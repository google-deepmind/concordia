# Copyright 2024 DeepMind Technologies Limited.
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

"""HTTP server for real-time simulation visualization.

This module provides a server that serves the visualization UI and broadcasts
simulation updates via Server-Sent Events (SSE).
"""

import http.server
import json
import queue
import socketserver
import sys
import threading
from typing import Any

from concordia.environment import step_controller as step_controller_lib


class SimulationServer:
  """HTTP server for real-time simulation control and visualization.

  Features:
    - Serves static HTML visualization
    - Server-Sent Events endpoint for real-time step updates
    - POST endpoints for play/pause/step control
  """

  def __init__(
      self,
      port: int = 8080,
      html_content: str = '',
  ):
    """Initialize the simulation server.

    Args:
      port: Port to serve on.
      html_content: Static HTML content to serve at the root.
    """
    self._port = port
    self._html_content = html_content
    self._step_controller = step_controller_lib.StepController(
        start_paused=True
    )
    self._server_sent_events_queues: list[queue.Queue[str]] = []
    self._server_sent_events_lock = threading.Lock()
    self._current_step_data: dict[str, Any] = {}
    self._cached_entity_info: dict[str, Any] | None = None
    self._simulation: Any = None
    self._server: socketserver.TCPServer | None = None
    self._server_thread: threading.Thread | None = None
    print(f'[SERVER INIT] SimulationServer initialized on port {port}')
    sys.stdout.flush()

  @property
  def step_controller(self) -> step_controller_lib.StepController:
    """Get the step controller for use by the simulation."""
    return self._step_controller

  @property
  def html_content(self) -> str:
    """Get the HTML content to serve."""
    return self._html_content

  @property
  def current_step_data(self) -> dict[str, Any]:
    """Get the current step data."""
    return self._current_step_data

  @property
  def server_sent_events_lock(self) -> threading.Lock:
    """Get the lock for thread-safe Server-Sent Events queue access."""
    return self._server_sent_events_lock

  @property
  def server_sent_events_queues(self) -> list[queue.Queue[str]]:
    """Get the list of Server-Sent Events client queues."""
    return self._server_sent_events_queues

  @property
  def cached_entity_info(self) -> dict[str, Any] | None:
    """Get cached entity info for new SSE clients."""
    return self._cached_entity_info

  def set_html_content(self, html_content: str) -> None:
    """Update the HTML content to serve."""
    self._html_content = html_content

  def set_simulation(self, simulation: Any) -> None:
    """Set the simulation instance for dynamic state editing.

    This must be called after the Simulation object is created so that
    the server can forward edit requests to it.

    Args:
      simulation: The Simulation instance.
    """
    self._simulation = simulation

  @property
  def simulation(self) -> Any:
    """Get the simulation instance."""
    return self._simulation

  def broadcast_step(self, step_data: step_controller_lib.StepData) -> None:
    """Broadcast step data to all connected Server-Sent Events clients.

    Args:
      step_data: The step data to broadcast.
    """
    self._current_step_data = {
        'step': step_data.step,
        'acting_entity': step_data.acting_entity,
        'action': step_data.action,
        'entity_actions': step_data.entity_actions,
        'entity_logs': step_data.entity_logs,
        'game_master': step_data.game_master,
    }
    message = f'data: {json.dumps(self._current_step_data)}\n\n'
    with self._server_sent_events_lock:
      dead_queues = []
      for q in self._server_sent_events_queues:
        try:
          q.put_nowait(message)
        except queue.Full:
          dead_queues.append(q)
      for q in dead_queues:
        self._server_sent_events_queues.remove(q)

  def broadcast_completion(self) -> None:
    """Broadcast simulation completion to all connected SSE clients."""
    completion_data = {
        'completion': True,
        'message': 'Simulation completed!',
    }
    message = f'data: {json.dumps(completion_data)}\n\n'
    with self._server_sent_events_lock:
      for q in self._server_sent_events_queues:
        try:
          q.put_nowait(message)
        except queue.Full:
          pass

  def broadcast_entity_info(self, checkpoint_data: dict[str, Any]) -> None:
    """Broadcast entity component info to all connected SSE clients.

    This sends the initial checkpoint data (containing component metadata)
    so the inspector panel can display component information in serve mode.

    Args:
      checkpoint_data: The checkpoint data from sim.make_checkpoint_data().
    """
    entity_info = {
        'entity_info': True,
        'entities': checkpoint_data.get('entities', {}),
        'game_masters': checkpoint_data.get('game_masters', {}),
    }
    self._cached_entity_info = entity_info
    message = f'data: {json.dumps(entity_info)}\n\n'
    with self._server_sent_events_lock:
      for q in self._server_sent_events_queues:
        try:
          q.put_nowait(message)
        except queue.Full:
          pass

  def _create_handler(self):
    """Create a request handler class with access to server state."""
    server = self

    class Handler(http.server.BaseHTTPRequestHandler):
      """HTTP request handler for simulation server."""

      def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
        """Suppress HTTP request logging."""
        del format, args  # Unused

      def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handle GET requests for HTML, Server-Sent Events, status, and commands."""
        print(f'[SERVER] GET request received: {self.path}')

        sys.stdout.flush()
        if self.path == '/':
          self._serve_html()
        elif self.path == '/events':
          self._serve_server_sent_events()
        elif self.path == '/status':
          self._serve_status()
        # GET-based command endpoints for testing
        elif self.path == '/cmd/step':
          print('[SERVER] GET /cmd/step - calling step()')
          sys.stdout.flush()
          server.step_controller.step()
          self._send_json({'status': 'stepping', 'method': 'GET'})
        elif self.path == '/cmd/play':
          print('[SERVER] GET /cmd/play - calling play()')
          sys.stdout.flush()
          server.step_controller.play()
          self._send_json({'status': 'playing', 'method': 'GET'})
        elif self.path == '/cmd/pause':
          print('[SERVER] GET /cmd/pause - calling pause()')
          sys.stdout.flush()
          server.step_controller.pause()
          self._send_json({'status': 'paused', 'method': 'GET'})
        else:
          self.send_error(404)

      def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle POST requests for simulation control commands."""
        if self.path == '/play':
          print('[SERVER] Calling step_controller.play()...')
          server.step_controller.play()
          print('[SERVER] play() completed, sending response...')
          self._send_json({'status': 'playing'})
          print('[SERVER] Response sent for /play')
        elif self.path == '/pause':
          print('[SERVER] Calling step_controller.pause()...')
          server.step_controller.pause()
          print('[SERVER] pause() completed, sending response...')
          self._send_json({'status': 'paused'})
          print('[SERVER] Response sent for /pause')
        elif self.path == '/step':
          print('[SERVER] Calling step_controller.step()...')
          server.step_controller.step()
          print('[SERVER] step() completed, sending response...')
          self._send_json({'status': 'stepping'})
          print('[SERVER] Response sent for /step')
        elif self.path == '/stop':
          print('[SERVER] Calling step_controller.stop()...')
          server.step_controller.stop()
          print('[SERVER] stop() completed, sending response...')
          self._send_json({'status': 'stopped'})
          print('[SERVER] Response sent for /stop')
        elif self.path == '/cmd/set_component_state':
          self._handle_set_component_state()
        else:
          print(f'[SERVER] Unknown path: {self.path}')
          self.send_error(404)

      def _serve_html(self) -> None:
        """Serve the visualization HTML."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(server.html_content.encode('utf-8'))

      def _serve_status(self) -> None:
        """Serve current simulation status."""
        status = {
            'is_running': server.step_controller.is_running,
            'is_paused': server.step_controller.is_paused,
            'current_step': server.current_step_data.get('step', 0),
        }
        self._send_json(status)

      def _serve_server_sent_events(self) -> None:
        """Serve Server-Sent Events stream."""
        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        server_sent_events_queue: queue.Queue[str] = queue.Queue(maxsize=100)
        with server.server_sent_events_lock:
          server.server_sent_events_queues.append(server_sent_events_queue)

        if server.current_step_data:
          initial = f'data: {json.dumps(server.current_step_data)}\n\n'
          self.wfile.write(initial.encode('utf-8'))
          self.wfile.flush()

        # Send cached entity info if available
        if server.cached_entity_info:
          entity_info_msg = (
              f'data: {json.dumps(server.cached_entity_info)}\n\n'
          )
          self.wfile.write(entity_info_msg.encode('utf-8'))
          self.wfile.flush()

        try:
          while True:
            try:
              message = server_sent_events_queue.get(timeout=30)
              self.wfile.write(message.encode('utf-8'))
              self.wfile.flush()
            except queue.Empty:
              self.wfile.write(b': keepalive\n\n')
              self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
          pass
        finally:
          with server.server_sent_events_lock:
            if server_sent_events_queue in server.server_sent_events_queues:
              server.server_sent_events_queues.remove(server_sent_events_queue)

      def _handle_set_component_state(self) -> None:
        """Handle POST /cmd/set_component_state for dynamic editing."""
        if not server.step_controller.is_paused:
          self._send_json({
              'status': 'error',
              'message': 'Simulation must be paused to edit state.',
          })
          return

        if server.simulation is None:
          self._send_json({
              'status': 'error',
              'message': 'No simulation instance available.',
          })
          return

        try:
          content_length = int(self.headers.get('Content-Length', 0))
          body = self.rfile.read(content_length)
          data = json.loads(body.decode('utf-8'))

          entity_name = data['entity_name']
          component_name = data['component_name']
          key = data['key']
          value = data['value']

          server.simulation.set_component_dynamic_state(
              entity_name=entity_name,
              component_name=component_name,
              key=key,
              value=value,
          )

          # Broadcast updated entity info so inspector refreshes
          checkpoint_data = server.simulation.make_checkpoint_data()
          server.broadcast_entity_info(checkpoint_data)

          print(
              f'[SERVER] Updated dynamic state: {entity_name}.'
              f'{component_name}.{key}'
          )
          self._send_json({
              'status': 'ok',
              'message': f'Updated {entity_name}.{component_name}.{key}',
          })
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
          print(f'[SERVER] Error setting component state: {e}')
          self._send_json({'status': 'error', 'message': str(e)})

      def _send_json(self, data: dict[str, Any]) -> None:
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    return Handler

  def start(self) -> None:
    """Start the HTTP server in a background thread."""
    handler = self._create_handler()
    self._server = socketserver.ThreadingTCPServer(('', self._port), handler)
    self._server.allow_reuse_address = True
    self._server_thread = threading.Thread(target=self._server.serve_forever)
    self._server_thread.daemon = True
    self._server_thread.start()
    print(f'Simulation server running at http://localhost:{self._port}')

  def stop(self) -> None:
    """Stop the HTTP server."""
    if self._server:
      self._server.shutdown()
      self._server = None
    if self._server_thread:
      self._server_thread.join(timeout=5)
      self._server_thread = None
