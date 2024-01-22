# Copyright 2023 DeepMind Technologies Limited.
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


"""Functions to convert python objects to HTML."""

import html

HTML_HEAD = """
  <!DOCTYPE html>
  <html>
  <head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
  body {font-family: Arial;}

  /* Style the tab */
  .tab {
    overflow: hidden;
    border: 1px solid #ccc;
    background-color: #f1f1f1;
  }

  /* Style the buttons inside the tab */
  .tab button {
    background-color: inherit;
    float: left;
    border: none;
    outline: none;
    cursor: pointer;
    padding: 14px 16px;
    transition: 0.3s;
    font-size: 17px;
  }

  /* Change background color of buttons on hover */
  .tab button:hover {
    background-color: #ddd;
  }

  /* Create an active/current tablink class */
  .tab button.active {
    background-color: #ccc;
  }

  /* Style the tab content */
  .tabcontent {
    display: none;
    padding: 6px 12px;
    border: 1px solid #ccc;
    border-top: none;
  }
  li {border: 2px solid black;}
  </style>
  </head>
  <body>
  """

HTML_TAIL = """
  <script>
  function openTab(evt, cityName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(cityName).style.display = "block";
    evt.currentTarget.className += " active";
  }
  </script>
  </body>
  </html>
  """


class HTMLWriter:
  """Class to write to HTML."""

  def __init__(self):
    self.html = ""

  def write(self, text):
    """Adds text to the HTML."""
    self.html += text

  def render(self):
    """Returns the HTML."""
    return self.html


class PythonObjectToHTMLConverter:
  """Class to convert python objects to HTML."""

  def __init__(self, python_object):
    self.python_object = python_object
    self.html_writer = HTMLWriter()

  def convert(self):
    self._convert_python_object(self.python_object)
    return self.html_writer.render()

  def _convert_python_object(self, python_object):
    """Converts a python object to HTML."""
    if isinstance(python_object, str):
      self.html_writer.write(html.escape(python_object))

    elif isinstance(python_object, list):
      for item in python_object:
        self._convert_python_object(item)
        self.html_writer.write("<br />")

    elif isinstance(python_object, dict):
      self.html_writer.write("<details>")

      if "date" in python_object.keys():
        self.html_writer.write("<summary>")
        self._convert_python_object(python_object["date"])
        if "Summary" in python_object.keys():
          self._convert_python_object("  " + python_object["Summary"])
        self.html_writer.write("</summary>")
      elif "Summary" in python_object.keys():
        self.html_writer.write("<summary>")
        self._convert_python_object("  " + python_object["Summary"])
        self.html_writer.write("</summary>")
      elif "Name" in python_object.keys():
        self.html_writer.write("<summary>")
        self._convert_python_object(python_object["Name"])
        self.html_writer.write("</summary>")

      for key, value in python_object.items():
        if key != "date" and key != "Summary":
          self.html_writer.write("<b><ul>")
          self._convert_python_object(key)
          self.html_writer.write("</b>")
          self.html_writer.write("<li>")
          self._convert_python_object(value)
          self.html_writer.write("</li></ul>")

      self.html_writer.write("</details>")
    else:
      self.html_writer.write(str(python_object))


def finalise_html(html_code):
  return HTML_HEAD + html_code + HTML_TAIL


def combine_html_pages(
    html_pages, tab_names, summary="", title="Experiment logs"
):
  """Combines multiple HTML pages into a single HTML page with tabs."""
  html_code = ""
  html_code += f"""<h2>{title}</h2>
  <p>{summary}</p>
  <p>Click on the buttons to see the detailed logs:</p>

  <div class="tab">
  """

  for tab_name in tab_names:
    html_code += (
        '<button class="tablinks" onclick="openTab(event,'
        f" '{tab_name}')\">{tab_name}</button>\n"
    )

  html_code += "</div>\n"

  for i, html_page in enumerate(html_pages):
    html_code += (
        f'<div id="{tab_names[i]}" class="tabcontent">' + html_page + "</div>\n"
    )

  return html_code

