# LLLM (Lamphear Large Language Model)

This is a project (mostly just for my own education and exploration) to build an LLM from scratch.

To get this up and running, start by installing the build dependencies (if you don't have them installed already):

1. Install `asdf` and update your shell config for asdf (i.e. add `. /usr/local/opt/asdf/libexec/asdf.sh` to your shell’s rc file)
2. Install `asdf-direnv` following the steps here: https://github.com/asdf-community/asdf-direnv
3. Follow [these instructions](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) to create a kaggle.json and put the resulting kaggle.json file in the root directory of this project.

Then run the remaining commands from the root directory of this project:
* `asdf plugin-add python`
* `asdf plugin-add poetry`
* `asdf install`
* `direnv allow`
* `poetry install`

Then you can automatically train the model and generate text by running `python main.py` from the root directory.