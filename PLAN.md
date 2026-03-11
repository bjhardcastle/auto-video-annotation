A library for loading a local video, extracting a subset of frames that capture a wide variety of content, then finding coordinates of key points in each frame. Key points should be supplied as a list of descriptive names so that an LLM can infer their expected position. 

Constraints:
- use anthropic's python SDK for LLM interactions
- use dotenv to load API keys
- use pydantic-settings for CLI tool
- read key points list from a json file containing a list of descriptive names
- output annotations file in json records format with the following structure:
  - frame number
  - key point name
  - x coordinate
  - y coordinate
- save a folder of annotated frames with key points overlaid for visual verification