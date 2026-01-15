#!/bin/sh

cd /app/data-science || exit 1
git fetch
OUTPUT=$(git status)

# if [ "$(git rev-list --count HEAD..@{upstream})" -gt 0 ]; then
if echo "$OUTPUT" | grep -q "Your branch is behind"; then
    git pull || exit 1
    pip install -r requirements.txt || exit 1
fi
python ./src/app.py 

