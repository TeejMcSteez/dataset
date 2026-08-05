#!/bin/bash

if [ "$(uname -s)" != "Linux" ]; then
    echo "script is only made to run on linux"
fi

if command -v python; then
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
else
    echo "Python is required please install"
fi
