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

"""Real Chromium UI, explicitly mocked speech APIs; no audio or model calls."""

import json
import pathlib
import threading
from unittest import mock

from concordia.examples.astral_canticle import human_io
from concordia.examples.bellwether import game_service
from concordia.examples.bellwether.multiplayer_browser_test import join
from concordia.examples.bellwether.multiplayer_test import approve
from concordia.examples.bellwether.multiplayer_test import envelope
from concordia.examples.bellwether.multiplayer_test import hosted_fixture  # pylint: disable=unused-import
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
from concordia.utils import simulation_server
import pytest

pw = pytest.importorskip('playwright.sync_api')

MOCK_SPEECH = """
window.voiceTest={starts:0, installs:0, cancelled:0, utterances:[], available:'available'};
class Recognition {
 get processLocally(){return this.local===true}
 set processLocally(value){this.local=value}
 static async available(options){voiceTest.options=options;return voiceTest.available}
 static async install(){voiceTest.installs++;return false}
 start(){voiceTest.starts++;voiceTest.rec=this;if(!this.processLocally)throw Error('Remote!')}
 stop(){this.onend?.()}
 abort(){voiceTest.aborted=true}
}
Object.defineProperty(window,'SpeechRecognition',{value:Recognition,configurable:true});
class Synthesis extends EventTarget {
 getVoices(){return [
  {name:'Remote',lang:'en-US',localService:false},
  ...(voiceTest.cloudOnly?[]:[{name:'Local',lang:'en-US',localService:true}])
 ]}
 speak(utterance){voiceTest.utterances.push({text:utterance.text,voice:utterance.voice.name})}
 cancel(){voiceTest.cancelled++}
}
Object.defineProperty(window,'speechSynthesis',{value:new Synthesis(),configurable:true});
Object.defineProperty(window,'SpeechSynthesisUtterance',{value:class {
 constructor(text){this.text=text}
},configurable:true});
window.finalTranscript=text=>{
 const result=[{transcript:text}];result.isFinal=true;
 voiceTest.rec.onresult?.({resultIndex:0,results:[result]});
};
"""


def wait_for_input(inbox, request):
  try:
    inbox(request)
  except human_io.InputClosed:
    pass  # Expected shutdown of a synthetic, deliberately unanswered turn.


@pytest.fixture(name='voice_host')
def voice_host_fixture(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation in voice UI tests'),
  ):
    game = game_service.Game(tmp_path)
    game.world.emit(
        'speech', 'The shelter keeper has arrived.', ['Coordinator']
    )
    game.world.emit('speech', 'SECRET_OTHER_ROLE', ['Nell'])
    request = human_input.HumanInputRequest(
        request_id='voice-input',
        entity_name='Coordinator',
        action_spec=entity_lib.free_action_spec(call_to_action='Your action'),
        contexts={},
        context='Your available text.',
    )
    reader = threading.Thread(target=wait_for_input, args=(game.inbox, request))
    server = simulation_server.SimulationServer(
        port=0,
        operation_service=game.operations,
        audience='player',
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
    )
    server.start()
    reader.start()
    try:
      yield game, f'http://127.0.0.1:{server.bound_port}'
    finally:
      game.close()
      server.stop()
      reader.join(2)


def test_editable_transcript_local_reading_and_cancellation(voice_host):
  _, url = voice_host
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    page = browser.new_page(
        viewport={'width': 360, 'height': 800}, is_mobile=True
    )
    page.add_init_script(MOCK_SPEECH)
    sent, errors = [], []
    page.on(
        'request',
        lambda r: sent.append(r.url) if '/dispatch' in r.url else None,
    )
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url)
    pw.expect(page.locator('#act')).to_be_enabled()
    page.locator('#voice summary').click()
    assert page.evaluate('voiceTest.starts') == 0
    assert page.evaluate('voiceTest.utterances') == []
    page.check('#voice-enabled')
    page.fill('#action', 'My typed proposal.')
    page.click('#dictate')
    pw.expect(page.locator('#voice-status')).to_contain_text('Listening')
    assert page.evaluate('voiceTest.options.processLocally') is True
    page.fill('#action', 'My typed proposal, edited during dictation.')
    page.evaluate(
        "finalTranscript('Unedited words <img src=x onerror=alert(1)>')"
    )
    assert 'Unedited words' in page.locator('#transcript').input_value()
    assert 'Unedited' not in page.locator('#action').input_value()
    page.fill('#transcript', 'Edited speech 🌊')
    page.click('#finish-dictation')
    page.click('#use-transcript')
    assert page.locator('#action').input_value() == (
        'My typed proposal, edited during dictation. Edited speech 🌊'
    )
    assert not sent
    page.click('#listen')
    assert page.evaluate('voiceTest.utterances') == [
        {'text': 'The shelter keeper has arrived.', 'voice': 'Local'}
    ]
    page.click('#stop-audio')
    assert page.evaluate('voiceTest.cancelled') == 1
    page.click('#dictate')
    page.evaluate('() => {window.late=voiceTest.rec.onresult}')
    page.click('#cancel-dictation')
    page.evaluate("""() => {
          const r=[{transcript:'LATE_PRIVATE'}];
          r.isFinal=true;late({resultIndex:0,results:[r]});
        }""")
    assert page.locator('#transcript').input_value() == ''
    assert 'LATE_PRIVATE' not in page.locator('#action').input_value()
    page.click('#dictate')
    page.evaluate("voiceTest.rec.onerror({error:'not-allowed'})")
    pw.expect(page.locator('#voice-status')).to_contain_text(
        'permission was declined'
    )
    page.evaluate(
        '() => {SpeechRecognition.available=()=>new'
        ' Promise(r=>voiceTest.resolve=r)}'
    )
    starts = page.evaluate('voiceTest.starts')
    page.click('#dictate')
    page.click('#cancel-dictation')
    page.evaluate("voiceTest.resolve('available')")
    assert page.evaluate('voiceTest.starts') == starts
    assert page.locator('img').count() == 0
    assert page.evaluate('document.documentElement.scrollWidth') <= 360
    page.reload()
    pw.expect(page.locator('#act')).to_be_enabled()
    assert not page.locator('#voice-enabled').is_checked()
    assert 'Edited speech' in page.locator('#action').input_value()
    assert not errors
    browser.close()


@pytest.mark.parametrize('case', ['unsupported', 'downloadable', 'remote-only'])
def test_unavailable_voice_keeps_text_without_remote_fallback(voice_host, case):
  _, url = voice_host
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    page = browser.new_page()
    page.add_init_script(MOCK_SPEECH)
    page.goto(url)
    pw.expect(page.locator('#act')).to_be_enabled()
    page.locator('#voice summary').click()
    if case == 'unsupported':
      page.evaluate('delete SpeechRecognition.prototype.processLocally')
    elif case == 'downloadable':
      page.evaluate("voiceTest.available='downloadable'")
    else:
      page.evaluate('voiceTest.cloudOnly=true')
    page.check('#voice-enabled')
    page.fill('#action', 'wait')
    if case == 'remote-only':
      pw.expect(page.locator('#listen')).to_be_disabled()
    else:
      page.click('#dictate')
      pw.expect(page.locator('#voice-status')).to_contain_text(
          'Please type instead'
      )
    assert page.locator('#action').input_value() == 'wait'
    assert page.evaluate('voiceTest.starts') == 0
    assert page.evaluate('voiceTest.installs') == 0
    assert page.evaluate('voiceTest.utterances') == []
    browser.close()


def test_role_revocation_cancels_audio_and_late_transcripts(hosted):
  game, _, url = hosted
  game.world.data['agenda'] = [{
      'name': 'Nell',
      'purpose': 'request: reserve',
      'audience': ['Coordinator', 'Nell'],
      'watch': 0,
  }]
  game.world.emit('speech', 'ONLY_NELL_OBSERVATION', ['Nell'])
  request = human_input.HumanInputRequest(
      request_id='nell-voice',
      entity_name='Nell',
      action_spec=entity_lib.free_action_spec(call_to_action='Respond'),
      contexts={},
      context='Only Nell context.',
  )
  reader = threading.Thread(
      target=wait_for_input, args=(game.nell_inbox, request)
  )
  reader.start()
  try:
    with pw.sync_playwright() as runner:
      browser = runner.chromium.launch()
      page = browser.new_page()
      page.add_init_script(MOCK_SPEECH)
      join(page, url, 'N', 'Nell')
      row = approve(game, 'N')
      pw.expect(page.locator('#act')).to_be_enabled()
      page.locator('#voice summary').click()
      page.check('#voice-enabled')
      page.click('#dictate')
      page.evaluate('() => {window.late=voiceTest.rec.onresult}')
      page.click('#listen')
      game.operations.dispatch(
          'developer', envelope(game, 'session.revoke', {'request_id': row})
      )
      pw.expect(page.locator('#join')).to_be_visible()
      assert page.evaluate('voiceTest.aborted') is True
      page.evaluate("""() => {
          const r=[{transcript:'REVOKED'}];
          r.isFinal=true;late({resultIndex:0,results:[r]});
        }""")
      assert page.locator('#transcript').input_value() == ''
      assert 'ONLY_NELL_OBSERVATION' not in page.content()
      assert page.evaluate('voiceTest.cancelled') == 1
      assert page.evaluate('voiceTest.utterances') == [
          {'text': 'ONLY_NELL_OBSERVATION', 'voice': 'Local'}
      ]
      browser.close()
  finally:
    game.nell_inbox.finish('done')
    reader.join(2)


def test_actual_browser_capabilities_do_not_start_audio(voice_host, tmp_path):
  _, url = voice_host
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    page = browser.new_page(
        viewport={'width': 412, 'height': 820}, is_mobile=True
    )
    requests = []
    page.on('request', lambda r: requests.append(r.url))
    page.goto(url)
    pw.expect(page.locator('#act')).to_be_enabled()
    page.locator('#voice summary').click()
    data = page.evaluate("""() => {
      const C=window.SpeechRecognition||window.webkitSpeechRecognition;
      return {
        recognition:!!C,onDeviceProperty:!!C&&('processLocally' in C.prototype),
        availabilityCheck:typeof C?.available==='function',
        localVoices:window.speechSynthesis?.getVoices().filter(v=>v.localService).length||0
      };
    }""")
    data['browser'] = browser.version
    (tmp_path / 'actual-capabilities.json').write_text(
        json.dumps(data, indent=2)
    )
    assert not page.locator('#voice-enabled').is_checked()
    pw.expect(page.locator('#dictate')).to_be_disabled()
    page.fill('#action', 'My text fallback.')
    assert page.locator('#action').input_value() == 'My text fallback.'
    assert all(r.startswith(url) for r in requests)
    browser.close()
