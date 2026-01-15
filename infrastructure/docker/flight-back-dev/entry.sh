#!/bin/sh

cd /flight-on-time-project/back-end || exit 1
git fetch
OUTPUT=$(git status)

# if [ "$(git rev-list --count HEAD..@{upstream})" -gt 0 ]; then
if echo "$OUTPUT" | grep -q "Your branch is behind"; then
    git pull origin dev
    mvn clean package || exit 1
fi
#java -Dspring.profiles.active=prod \
java \
-jar ./target/flightontime-0.0.1-MVP-ALPHA.jar

