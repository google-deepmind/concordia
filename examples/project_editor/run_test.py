# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example setup and export checks; no simulation/model execution."""

import copy
import json
from pathlib import Path
import sys
from unittest import mock

from concordia.agents import entity_agent
from concordia.environment import step_controller
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
import pytest

from examples.project_editor import run
from examples.project_editor import template


@pytest.fixture(autouse=True)
def isolate_together_credentials(monkeypatch):
  """Tests use only synthetic provider credentials."""
  monkeypatch.delenv('TOGETHER_API_KEY', raising=False)
  monkeypatch.delenv('TOGETHER_AI_API_KEY', raising=False)


@pytest.mark.parametrize('live', [False, True])
@pytest.mark.parametrize(
    ('port', 'bound_port', 'public_origin', 'expected_url'),
    [
        (8081, 8081, None, 'http://127.0.0.1:8081/'),
        (0, 49152, None, 'http://127.0.0.1:49152/'),
        (
            8081,
            8081,
            'https://editor.example:10000',
            'https://editor.example:10000/',
        ),
    ],
)
def test_main_advertises_browser_origin_without_starting_listener(
    port,
    bound_port,
    public_origin,
    expected_url,
    capsys,
    live,
):
  server = mock.Mock(spec=run.simulation_server.SimulationServer)
  server.bound_port = bound_port
  args = ['project_editor', '--port', str(port)]
  if live:
    args += ['--model-backend', 'together_ai', '--model-name', 'provider/model']
  if public_origin:
    args += ['--public-origin', public_origin]
  with (
      mock.patch.object(sys, 'argv', args),
      mock.patch.object(run, 'create_editor', return_value=server) as create,
      mock.patch.object(
          run, 'build', side_effect=AssertionError('No simulation')
      ),
      mock.patch.object(run.time, 'sleep', side_effect=KeyboardInterrupt),
  ):
    run.main()
  assert create.call_args.kwargs['port'] == port
  assert create.call_args.kwargs['public_origin'] == public_origin
  server.start.assert_called_once_with()  # Mock only; never binds a socket.
  server.stop.assert_called_once_with()
  output = capsys.readouterr().out
  mode = 'Live' if live else 'Mock'
  assert f'{mode} editor: {expected_url}' in output
  assert create.call_args.kwargs['model_selection'] == (
      run.ModelSelection('together_ai', 'provider/model')
      if live
      else run.ModelSelection()
  )
  if public_origin:
    assert 'http://127.0.0.1:' not in output


def test_open_builds_preview_without_running():
  with (
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError,
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError,
      ),
  ):
    server = run.create_editor(port=0)
    assert not server.is_serving
    assert server.simulation is None
    service = server.operation_service
    assert service is not None
    state = service.snapshot('developer')['result']
    assert state['state'] == 'ready'
    assert 'Free mock' in server.html_content
    fields = state['definition']['inspector']['conversation']
    assert 'acting_order' in fields
    assert state['document']['max_steps'] == 40
    assert state['document']['schema_version'] == 4
    assert 'scenes' not in state['document']
    assert (
        'Goal'
        in state['definition']['entities']['entity_1']['component_info'][
            'context_components'
        ]
    )


@pytest.mark.parametrize('delay', [-1, float('inf'), float('nan'), 6])
def test_pacing_is_finite_and_bounded(delay):
  with pytest.raises(ValueError, match='delay'):
    run.create_editor(step_delay=delay)


def test_each_export_preserves_definition_and_previous_log(tmp_path: Path):
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  log = mock.Mock()
  log.to_json.return_value = '{"fixture":true}'
  log.to_html.return_value = '<p>Fixture</p>'
  first = run.save_result(registry, document, log, tmp_path)
  second = run.save_result(registry, document, log, tmp_path)
  assert first != second
  for directory in (first, second):
    assert (
        json.loads((directory / 'initial-project.json').read_text()) == document
    )
    assert (directory / 'log.json').read_text() == '{"fixture":true}'


def test_builder_and_legacy_documents_remain_distinct():
  registry = template.registry()
  builder = registry.default_document(template.TEMPLATE_KEY)
  assert builder['schema_version'] == 4
  assert [x['prototype'] for x in builder['instances']] == [
      'alice',
      'bob',
      'conversation',
  ]
  assert [
      entry['key']
      for entry in registry.catalog(builder)
      if not entry['instance']['prototype'].startswith('installed:')
  ] == [
      'minimal',
      'basic',
      'dialogic',
      'alice',
      'bob',
      'conversation',
  ]
  discovered = {entry['key'] for entry in registry.catalog(builder)}
  assert 'entity.rational.Entity' in discovered
  assert 'contrib.game_master.space_ship.GameMaster' in discovered
  old_builder = registry.default_document(template.STRUCTURAL_TEMPLATE_KEY)
  assert old_builder['schema_version'] == 2
  assert registry.loads(registry.dumps(old_builder)) == old_builder
  for key in (template.LEGACY_TEMPLATE_KEY, template.PREVIOUS_TEMPLATE_KEY):
    legacy = registry.default_document(key)
    assert legacy['schema_version'] == 1
    assert registry.loads(registry.dumps(legacy)) == legacy
    assert registry.catalog(legacy) == []
    assert all('prototype' not in item for item in legacy['instances'])


def test_builder_validates_each_duplicated_gm():
  registry = template.registry()
  document = registry.default_document(template.STRUCTURAL_TEMPLATE_KEY)
  gm = copy.deepcopy(document['instances'][2])
  gm['id'] = 'new-gm'
  gm['params']['name'] = 'New GM'
  gm['params']['acting_order'] = 'unsupported'
  document['instances'].append(gm)
  with pytest.raises(ValueError, match='new-gm'):
    registry.normalize(document)


@pytest.mark.parametrize(
    ('owner', 'kind'),
    [
        ('alice', 'constant'),
        ('alice', 'recent-observations'),
        ('bob', 'constant'),
        ('bob', 'recent-observations'),
    ],
)
def test_catalogue_components_reach_each_example_prefab(owner, kind):
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  spec = next(
      x for x in registry.component_catalog(document) if x['key'] == kind
  )
  params = dict(spec['defaults'])
  params['pre_act_label'] = 'Example context'
  params['state' if kind == 'constant' else 'history_length'] = (
      'Literal example 🎵' if kind == 'constant' else 4
  )
  document['components'] = [
      dict(
          id='example-context',
          instance=owner,
          type=kind,
          name='Example component',
          params=params,
      )
  ]
  with (
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError,
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError,
      ),
  ):
    simulation = run.build(
        registry.to_config(registry.loads(registry.dumps(document)))
    )
    name = next(
        x['params']['name'] for x in document['instances'] if x['id'] == owner
    )
    entity = next(
        x
        for x in [*simulation.get_entities(), *simulation.get_game_masters()]
        if x.name == name
    )
    assert isinstance(entity, entity_agent.EntityAgent)
    component = entity.get_component('authored_example-context')
    assert all(
        component.get_state()[key] == value for key, value in params.items()
    )
    order = entity.get_act_component().get_state()['component_order']
    assert isinstance(order, list)
    assert order[-1] == 'authored_example-context'


@pytest.mark.parametrize(
    ('steps', 'stop', 'reason'),
    [
        (10, False, 'Requested step limit reached (10).'),
        (
            2,
            False,
            'The game master ended the run before the requested step limit',
        ),
        (1, True, 'Stop requested through the run controls.'),
    ],
)
def test_runner_completion_reason_with_mocked_simulation(steps, stop, reason):
  fake = mock.Mock()
  fake.make_checkpoint_data.return_value = {'entities': {}, 'game_masters': {}}
  with (
      mock.patch.object(run, 'build', return_value=fake),
      mock.patch.object(run, 'save_result', return_value=Path('mock-result')),
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError,
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError,
      ),
  ):
    server = run.create_editor(step_delay=0)

    def mock_play(**kwargs):
      kwargs['step_callback'](
          step_controller.StepData(
              step=steps,
              acting_entity='Alice',
              action='Alice: Alice',
              entity_actions={'Alice': 'Alice: Alice'},
              entity_logs={},
          )
      )
      if stop:
        kwargs['step_controller'].stop()
      return mock.Mock()

    fake.play.side_effect = mock_play
    server.run_project(0)
    assert server._project_thread is not None
    server._project_thread.join(3)
    assert not server._project_thread.is_alive()
    assert server.get_project()['run']['message'].startswith(reason)
    assert server.get_project()['run']['status'] == (
        'stopped' if stop else 'completed'
    )


def test_new_personas_use_third_person_and_old_saved_text_is_preserved():
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  assert document['instances'][0]['params']['custom_instructions'].startswith(
      'Alice is'
  )
  assert document['instances'][1]['params']['goal'].startswith('Bob is')
  old = registry.default_document(template.LEGACY_TEMPLATE_KEY)
  text = old['instances'][0]['params']['custom_instructions']
  assert text.startswith('You are Alice')
  assert (
      registry.loads(registry.dumps(old))['instances'][0]['params'][
          'custom_instructions'
      ]
      == text
  )


def test_default_selection_uses_standard_no_model():
  assert isinstance(
      run.ModelSelection().create_model(), no_language_model.NoLanguageModel
  )
  assert 'Free mock' in run.ModelSelection().label


@pytest.mark.parametrize(
    'backend,name',
    [
        ('none', 'unused'),
        ('together_ai', None),
        ('together_ai', ' '),
        ('', None),
    ],
)
def test_invalid_model_selection(backend, name):
  with pytest.raises(ValueError):
    run.ModelSelection(backend, name)


def test_live_selection_uses_standard_factory_without_sampling():
  selection = run.ModelSelection('together_ai', 'provider/model')
  with mock.patch.object(
      run.language_models, 'language_model_setup'
  ) as factory:
    assert selection.create_model() is factory.return_value
  factory.assert_called_once_with(
      api_type='together_ai',
      model_name='provider/model',
      api_key=None,
      disable_language_model=False,
  )
  assert 'Free mock' not in selection.label
  assert 'together_ai' in selection.label
  assert 'provider/model' in selection.label


def test_live_preview_edit_save_never_initializes_provider():
  with (
      mock.patch.object(
          run.os, 'getenv', side_effect=AssertionError('No credential lookup')
      ),
      mock.patch.object(
          run.language_models,
          'language_model_setup',
          side_effect=AssertionError('No provider initialization'),
      ),
      mock.patch.object(
          generic.Simulation, 'play', side_effect=AssertionError('No execution')
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError('No sampling'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError('No sampling'),
      ),
  ):
    server = run.create_editor(
        model_selection=run.ModelSelection('together_ai', 'provider/model')
    )
    assert 'Live model configured' in server.html_content
    assert 'together_ai' in server.html_content
    assert 'provider/model' in server.html_content
    assert 'Free mock' not in server.html_content
    document = server.get_project()['document']
    document['instances'][0]['params']['goal'] = 'Alice wants a quiet evening.'
    saved = server.replace_project(template.registry().dumps(document), 0)
    assert saved['revision'] == 1
    assert saved['document'] == document
    assert not server.is_serving
    assert server.simulation is None


def test_build_passes_supplied_model_to_standard_simulation():
  selected = mock.Mock()
  with mock.patch.object(run.generic, 'Simulation') as simulation:
    run.build(
        template.registry().to_config(
            template.registry().default_document(template.TEMPLATE_KEY)
        ),
        model=selected,
    )
  assert simulation.call_args.kwargs['model'] is selected


@pytest.mark.parametrize('requested', [None, 3, 40])
def test_editor_run_uses_selected_model_with_mock_execution(requested):
  selected = mock.Mock()
  fake = mock.Mock()
  fake.make_checkpoint_data.return_value = {'entities': {}, 'game_masters': {}}
  with (
      mock.patch.object(
          run.language_models, 'language_model_setup', return_value=selected
      ) as factory,
      mock.patch.object(run, 'build', return_value=fake) as build,
      mock.patch.object(run, 'save_result'),
  ):
    server = run.create_editor(
        step_delay=0,
        model_selection=run.ModelSelection('together_ai', 'provider/model'),
    )
    factory.assert_not_called()
    server.run_project(0, requested)
    assert server._project_thread is not None
    server._project_thread.join(3)
    assert not server._project_thread.is_alive()
    assert build.call_args.kwargs['model'] is selected
    factory.assert_called_once()
    fake.play.assert_called_once()
    assert fake.play.call_args.kwargs['max_steps'] == (
        10 if requested is None else requested
    )
    assert server.get_project()['run']['status'] == 'completed'


@pytest.mark.parametrize(
    'error',
    [
        ValueError('Unrecognized api_type: invalid'),
        ImportError('Install provider dependency'),
        ValueError('Provider configuration missing'),
    ],
)
def test_provider_setup_error_is_retained_without_mock_fallback(error):
  with (
      mock.patch.object(
          run.language_models, 'language_model_setup', side_effect=error
      ),
      mock.patch.object(
          generic.Simulation, 'play', side_effect=AssertionError('No execution')
      ),
  ):
    server = run.create_editor(
        model_selection=run.ModelSelection('invalid', 'model')
    )
    server.run_project(0)
    assert server._project_thread is not None
    server._project_thread.join(3)
    assert not server._project_thread.is_alive()
    result = server.get_project()
    assert result['run']['status'] == 'failed'
    assert result['run']['message'] == str(error)
    assert server.simulation is None
    assert result['document']['max_steps'] == 40


def test_headless_cli_selection_and_limit_without_execution(capsys):
  selected = mock.Mock()
  with (
      mock.patch.object(
          sys,
          'argv',
          [
              'editor',
              '--headless',
              '--model-backend',
              'together_ai',
              '--model-name',
              'provider/model',
          ],
      ),
      mock.patch.object(
          run.language_models, 'language_model_setup', return_value=selected
      ),
      mock.patch.object(run, 'build') as build,
      mock.patch.object(run, 'save_result') as save,
      mock.patch.object(
          run, 'create_editor', side_effect=AssertionError('No server')
      ),
  ):
    run.main()
  assert build.call_args.kwargs['model'] is selected
  assert build.call_args.args[0].default_max_steps == 40
  build.return_value.play.assert_called_once_with(max_steps=10)
  assert save.call_args.args[1]['max_steps'] == 40
  assert 'Live model configured' in capsys.readouterr().out


@pytest.mark.parametrize(
    'args',
    [
        ['--model-name', 'model'],
        ['--model-backend', 'together_ai'],
        ['--max-steps', '0'],
        ['--max-steps', '1001'],
    ],
)
def test_cli_rejects_invalid_selection_before_initialization(args):
  with (
      mock.patch.object(sys, 'argv', ['editor', *args]),
      mock.patch.object(run.language_models, 'language_model_setup') as factory,
      mock.patch.object(run, 'create_editor') as create,
      pytest.raises(SystemExit) as result,
  ):
    run.main()
  assert result.value.code == 2
  factory.assert_not_called()
  create.assert_not_called()


def test_headless_default_and_provider_failure_without_execution():
  with (
      mock.patch.object(sys, 'argv', ['editor', '--headless']),
      mock.patch.object(run, 'build') as build,
      mock.patch.object(run, 'save_result'),
  ):
    run.main()
  assert isinstance(
      build.call_args.kwargs['model'], no_language_model.NoLanguageModel
  )
  with (
      mock.patch.object(
          sys,
          'argv',
          [
              'editor',
              '--headless',
              '--model-backend',
              'unknown',
              '--model-name',
              'model',
          ],
      ),
      mock.patch.object(run, 'build') as build,
      pytest.raises(ValueError, match='Unrecognized api_type: unknown'),
  ):
    run.main()
  build.assert_not_called()


@pytest.mark.parametrize('sdk_key', [None, '', 'synthetic-sdk-key'])
@pytest.mark.parametrize('adapter_key', [None, 'synthetic-adapter-key'])
def test_together_sdk_environment_bridge(monkeypatch, sdk_key, adapter_key):
  if sdk_key is not None:
    monkeypatch.setenv('TOGETHER_API_KEY', sdk_key)
  if adapter_key is not None:
    monkeypatch.setenv('TOGETHER_AI_API_KEY', adapter_key)
  with mock.patch.object(
      run.language_models, 'language_model_setup'
  ) as factory:
    run.ModelSelection('together_ai', 'provider/model').create_model()
  factory.assert_called_once_with(
      api_type='together_ai',
      model_name='provider/model',
      api_key=sdk_key or None,
      disable_language_model=False,
  )
  # None leaves the existing adapter's TOGETHER_AI_API_KEY fallback intact.
  assert run.os.getenv('TOGETHER_AI_API_KEY') == adapter_key


@pytest.mark.parametrize('backend,name', [('none', None), ('openai', 'model')])
def test_other_backends_do_not_read_or_forward_together_key(backend, name):
  with (
      mock.patch.object(
          run.os,
          'getenv',
          side_effect=AssertionError('No Together credential lookup'),
      ),
      mock.patch.object(run.language_models, 'language_model_setup') as factory,
  ):
    run.ModelSelection(backend, name).create_model()
  assert factory.call_args.kwargs['api_key'] is None


def test_headless_clamps_to_imported_maximum_without_rewriting(tmp_path):
  document = template.registry().default_document(template.TEMPLATE_KEY)
  document['max_steps'] = 4
  project = tmp_path / 'project.json'
  project.write_text(json.dumps(document))
  with (
      mock.patch.object(
          sys, 'argv', ['editor', '--headless', '--project', str(project)]
      ),
      mock.patch.object(run, 'build') as build,
      mock.patch.object(run, 'save_result') as save,
  ):
    run.main()
  build.return_value.play.assert_called_once_with(max_steps=4)
  assert save.call_args.args[1] == document


@pytest.mark.parametrize(
    'flags',
    [
        [],
        ['--model-backend', 'other', '--model-name', 'm'],
        ['--model-backend', 'together_ai'],
        [
            '--model-backend',
            'together_ai',
            '--model-name',
            'm',
            '--step-delay',
            'nan',
        ],
        ['--model-backend', 'together_ai', '--model-name', 'm', '--port', '-1'],
        [
            '--model-backend',
            'together_ai',
            '--model-name',
            'm',
            '--public-origin',
            'http://wrong',
        ],
    ],
)
def test_prompt_validates_flags_before_input_or_server(flags):
  with (
      mock.patch.object(sys, 'argv', ['editor', '--prompt-api-key', *flags]),
      mock.patch.object(run.getpass, 'getpass') as prompt,
      mock.patch.object(run, 'create_editor') as create,
      pytest.raises(SystemExit) as error,
  ):
    run.main()
  assert error.value.code == 2
  prompt.assert_not_called()
  create.assert_not_called()


def test_prompt_rejects_non_tty_before_reading():
  with (
      mock.patch.object(sys.stdin, 'isatty', return_value=False),
      mock.patch.object(run.getpass, 'getpass') as prompt,
      pytest.raises(ValueError, match='local interactive terminal'),
  ):
    run.prompt_api_key()
  prompt.assert_not_called()


@pytest.mark.parametrize(
    'result',
    [
        '',
        '  ',
        EOFError(),
        KeyboardInterrupt(),
        OSError('synthetic-key-must-not-appear'),
    ],
)
def test_prompt_failure_is_clean_and_before_server(result, capsys):
  with (
      mock.patch.object(
          sys,
          'argv',
          [
              'editor',
              '--prompt-api-key',
              '--model-backend',
              'together_ai',
              '--model-name',
              'm',
          ],
      ),
      mock.patch.object(sys.stdin, 'isatty', return_value=True),
      mock.patch.object(
          run.getpass,
          'getpass',
          side_effect=result if isinstance(result, BaseException) else None,
          return_value=result,
      ),
      mock.patch.object(run, 'create_editor') as create,
      pytest.raises(SystemExit) as error,
  ):
    run.main()
  assert error.value.code == 2
  create.assert_not_called()
  assert 'synthetic-key-must-not-appear' not in capsys.readouterr().err


def test_getpass_warning_prevents_echo_fallback():
  fallback = mock.Mock()

  def fake_getpass(_):
    run.warnings.warn('Cannot control echo', run.getpass.GetPassWarning)
    fallback()

  with (
      mock.patch.object(sys.stdin, 'isatty', return_value=True),
      mock.patch.object(run.getpass, 'getpass', side_effect=fake_getpass),
      pytest.raises(ValueError, match='unavailable'),
  ):
    run.prompt_api_key()
  fallback.assert_not_called()


def test_prompted_selection_redaction_and_factory_forwarding():
  key = 'synthetic-local-input'
  selection = run.ModelSelection('together_ai', 'm', api_key=key)
  assert key not in repr(selection)
  assert key not in selection.label
  assert run.dataclasses.asdict(selection) == {
      'backend': 'together_ai',
      'model_name': 'm',
  }
  with (
      mock.patch.object(
          run.os, 'getenv', side_effect=AssertionError('Do not consult env')
      ),
      mock.patch.object(run.language_models, 'language_model_setup') as factory,
  ):
    selection.create_model()
  assert factory.call_args.kwargs['api_key'] == key
  with (
      mock.patch.object(
          run.language_models,
          'language_model_setup',
          side_effect=RuntimeError(key),
      ),
      pytest.raises(ValueError) as error,
  ):
    selection.create_model()
  assert key not in str(error.value)


def test_explicit_prompt_forwards_to_editor_in_memory_only(capsys):
  server = mock.Mock()
  server.bound_port = 8080
  key = 'synthetic-local-input'
  with (
      mock.patch.object(
          sys,
          'argv',
          [
              'editor',
              '--prompt-api-key',
              '--model-backend',
              'together_ai',
              '--model-name',
              'm',
          ],
      ),
      mock.patch.object(sys.stdin, 'isatty', return_value=True),
      mock.patch.object(
          run.os, 'putenv', side_effect=AssertionError('No environment writes')
      ),
      mock.patch.object(run.getpass, 'getpass', return_value=key) as prompt,
      mock.patch.object(run, 'create_editor', return_value=server) as create,
      mock.patch.object(run.time, 'sleep', side_effect=KeyboardInterrupt),
      mock.patch.object(run.language_models, 'language_model_setup') as factory,
  ):
    run.main()
  prompt.assert_called_once()
  factory.assert_not_called()
  selection = create.call_args.kwargs['model_selection']
  assert selection._runtime_api_key == key
  assert key not in repr(create.call_args)
  assert key not in ''.join(capsys.readouterr())


def test_absent_prompt_flag_never_reads_input():
  with (
      mock.patch.object(sys, 'argv', ['editor', '--headless']),
      mock.patch.object(
          run.getpass, 'getpass', side_effect=AssertionError('No prompt')
      ) as prompt,
      mock.patch.object(run, 'build'),
      mock.patch.object(run, 'save_result'),
  ):
    run.main()
  prompt.assert_not_called()


def test_prompted_key_not_in_preview_snapshot_or_document():
  key = 'synthetic-local-input'
  selection = run.ModelSelection('together_ai', 'm', api_key=key)
  with mock.patch.object(
      run.language_models,
      'language_model_setup',
      side_effect=AssertionError('No provider during preview'),
  ):
    server = run.create_editor(model_selection=selection)
  assert key not in server.html_content
  assert key not in json.dumps(server.get_project())
  assert server.operation_service is not None
  assert key not in json.dumps(server.operation_service.snapshot('developer'))


def test_public_cli_tutorial_against_actual_example(tmp_path, capsys):
  """Execute the published authoring block verbatim, never a simulation."""
  from concordia.command_line_interface import concordia_session
  from concordia.utils import session_commands_test

  readme = (
      Path(__file__).parents[2] / 'concordia/command_line_interface/README.md'
  ).read_text()
  start = readme.index('```text\nset alice params.goal') + len('```text\n')
  lines = readme[start : readme.index('```', start)].strip().splitlines()
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('No simulation'),
      ),
      mock.patch.object(
          run.simulation_server.SimulationServer,
          'start',
          side_effect=AssertionError('No listener'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError('No model'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError('No model'),
      ),
      mock.patch.object(
          run.ModelSelection,
          'create_model',
          side_effect=AssertionError('No provider initialization'),
      ),
  ):
    server = run.create_editor(output=tmp_path / 'runs', port=0)
    with (
        mock.patch('builtins.input', side_effect=[*lines, 'exit']),
        mock.patch.object(
            concordia_session.urllib.request,
            'urlopen',
            side_effect=session_commands_test.fake_http(server),
        ),
    ):
      assert (
          concordia_session.main([
              '--url',
              'http://fixture',
              'interactive',
              '--draft',
              str(tmp_path / 'draft.json'),
          ])
          == 0
      )
    assert 'scenes' not in server.get_project()['document']
    assert (
        server.get_project()['document']['components'][0]['instance']
        == 'Charlie'
    )
  assert 'Error:' not in capsys.readouterr().out
