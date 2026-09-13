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

"""Optional single-controller web transport. Importing never starts a game."""

import argparse
import contextlib
import logging
import pathlib
import threading

from concordia.contrib.language_models.ollama import ollama_model
from concordia.examples.astral_canticle import adventure
from concordia.examples.astral_canticle import human_io
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

_LOG = logging.getLogger(__name__)
_STATIC = pathlib.Path(__file__).with_name('static')


class Submission(BaseModel):
  request_id: str = Field(min_length=1, max_length=128)
  response: str = Field(min_length=1, max_length=8000)


def create_app(
    session: human_io.HumanSession,
    *,
    runner=None,
    allowed_hosts=('127.0.0.1', 'localhost'),
    root_path='',
) -> FastAPI:
  """Create a browser adapter; the optional runner owns one adventure process.

  There are no start/reset/lobby endpoints. Reloading cannot launch a second
  simulation. Bind to loopback and use a private authenticated reverse proxy
  such as Tailscale Serve. All permitted tailnet users share this one controller.
  """

  @contextlib.asynccontextmanager
  async def lifespan(app):
    del app
    thread = None
    if runner is not None:

      def run():
        try:
          runner()
        except human_io.InputClosed:
          pass
        except Exception:  # pylint: disable=broad-exception-caught
          _LOG.exception('Adventure stopped')
          session.finish(
              'The story paused unexpectedly. Your journal is safe; '
              'ask the host to check the adventure log.'
          )

      thread = threading.Thread(
          target=run, name='concordia-adventure', daemon=True
      )
      thread.start()
    yield
    session.finish('The adventure host has stopped. Your journal is saved.')
    if thread is not None:
      thread.join(timeout=1)

  app = FastAPI(
      lifespan=lifespan,
      docs_url=None,
      redoc_url=None,
      openapi_url=None,
      root_path=root_path,
  )
  app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

  @app.middleware('http')
  async def browser_boundary(request: Request, call_next):
    # ASGI root_path denotes the mount location. Starlette's StaticFiles needs
    # scope.path to include it, even when a reverse proxy stripped the prefix.
    # Normalize once so both stripping and preserving proxies work.
    path = request.scope['path']
    if root_path and path != root_path and not path.startswith(root_path + '/'):
      request.scope['path'] = root_path + path
      request.scope['raw_path'] = request.scope['path'].encode('utf-8')
    # The browser supplies Origin. Match host (including port), and require JSON
    # plus a custom header: a foreign page cannot issue a simple form POST.
    if request.method == 'POST':
      origin = request.headers.get('origin', '')
      host = request.headers.get('host', '')
      if origin not in (f'http://{host}', f'https://{host}'):
        return JSONResponse(
            {'detail': 'Same-origin requests only.'}, status_code=403
        )
      if request.headers.get('x-astral-client') != '1':
        return JSONResponse(
            {'detail': 'Missing client header.'}, status_code=403
        )
      if (
          request.headers.get('content-type', '').split(';')[0]
          != 'application/json'
      ):
        return JSONResponse({'detail': 'JSON required.'}, status_code=415)
      try:
        if int(request.headers.get('content-length', '0')) > 40000:
          return Response(status_code=413)
      except ValueError:
        return Response(status_code=400)
      # Enforce actual size too, including chunked requests.
      body = await request.body()
      if len(body) > 40000:
        return Response(status_code=413)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    )
    return response

  @app.get('/')
  def index():
    return FileResponse(_STATIC / 'index.html')

  @app.get('/api/state')
  def state():
    return session.snapshot()

  @app.post('/api/action')
  def submit(body: Submission):
    try:
      accepted = session.submit(body.request_id, body.response)
    except human_io.StaleRequest as exc:
      raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'accepted': accepted}

  @app.get('/api/journal')
  def journal():
    snapshot = session.snapshot()
    text = 'THE ASTRAL CANTICLE\n\n' + '\n\n'.join(
        ('> ' if entry['kind'] == 'action' else '') + entry['text']
        for entry in snapshot['entries']
    )
    return Response(
        text,
        media_type='text/plain',
        headers={
            'Content-Disposition': (
                'attachment; filename="astral-canticle-journal.txt"'
            )
        },
    )

  app.mount('/static', StaticFiles(directory=_STATIC), name='static')
  return app


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', default='llama3.2:3b')
  parser.add_argument('--role', choices=('player', 'gm'), default='player')
  parser.add_argument(
      '--player-prefab',
      choices=('minimal', 'basic'),
      default='minimal',
      help='Standard prefab for Ilyra; basic retains its LLM perceptions.',
  )
  parser.add_argument('--port', type=int, default=8767)
  parser.add_argument('--allowed-host', action='append', default=[])
  parser.add_argument('--root-path', default='')
  parser.add_argument('--max-steps', type=int, default=30)
  parser.add_argument(
      '--output', type=pathlib.Path, default=pathlib.Path('runs/human')
  )
  args = parser.parse_args()
  if not 1 <= args.max_steps <= 300:
    parser.error('--max-steps must be between 1 and 300')
  if not 1 <= args.port <= 65535:
    parser.error('--port must be between 1 and 65535')
  session = human_io.HumanSession(role=args.role)
  model = ollama_model.OllamaLanguageModel(model_name=args.model)
  app = create_app(
      session,
      runner=lambda: adventure.play(
          model,
          session,
          args.output,
          role=args.role,
          max_steps=args.max_steps,
          player_prefab=args.player_prefab,
      ),
      allowed_hosts=['127.0.0.1', 'localhost', *args.allowed_host],
      root_path=args.root_path,
  )
  uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
  main()
