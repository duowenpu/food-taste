#!/bin/bash
# Auto early-stop: stop scorer-train when the best val Spearman is >=2 evals old (no +0.002 improvement).
SOCK=/var/run/docker.sock; CID=02a5d53cf19783a6d8b2b2a64ebdccaa5ae637339d8e21ca2ebb5faecda1dc82
while :; do
  state=$(curl -s --unix-socket $SOCK http://localhost/containers/$CID/json | grep -o '"Running":[a-z]*')
  [ "$state" != '"Running":true' ] && { echo "train container no longer running; watcher exits"; break; }
  vals=$(curl -s --unix-socket $SOCK "http://localhost/containers/$CID/logs?stdout=true&stderr=true"         | tr -c '[:print:]\n' ' ' | grep -oE 'val Spearman [+-][0-9.]+' | awk '{print $3}')
  n=$(echo "$vals" | grep -c .)
  if [ "$n" -ge 3 ]; then
    best_idx=$(echo "$vals" | awk 'BEGIN{b=-2;bi=0} {i++; if($1>b+0.002){b=$1;bi=i}} END{print bi}')
    if [ $((n - best_idx)) -ge 2 ]; then
      echo "PLATEAU: best at eval #$best_idx of $n -> stopping scorer-train ($(date))"
      curl -s -X POST --unix-socket $SOCK "http://localhost/containers/$CID/stop?t=30" >/dev/null
      break
    fi
  fi
  echo "$(date '+%H:%M') evals=$n values: $(echo $vals | tr '\n' ' ')"
  sleep 600
done
