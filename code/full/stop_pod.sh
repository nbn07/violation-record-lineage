#!/usr/bin/env bash
# Stops this RunPod pod (the GPU stops billing; files in /workspace are kept).
#   bash stop_pod.sh --check   -> only checks that stopping will work
#   bash stop_pod.sh           -> stops the pod now
if [ "$1" == "--check" ]; then
  ok=1
  if [ -z "$RUNPOD_POD_ID" ]; then echo "MISSING: RUNPOD_POD_ID is not set (are you on a RunPod pod?)"; ok=0; else echo "OK: pod id $RUNPOD_POD_ID"; fi
  if command -v runpodctl >/dev/null 2>&1; then echo "OK: runpodctl is installed"; else echo "MISSING: runpodctl not found"; ok=0; fi
  if [ -n "$RUNPOD_API_KEY" ]; then echo "OK: an API key is available to the pod"; else echo "NOTE: RUNPOD_API_KEY not set; runpodctl may still work, otherwise stop the pod from the website"; fi
  [ $ok == 1 ] && echo "Auto-stop should work." || echo "Auto-stop may NOT work: remember to stop the pod yourself on the RunPod website."
  exit 0
fi
echo "Stopping pod $RUNPOD_POD_ID ..."
runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null && exit 0
runpodctl pod stop "$RUNPOD_POD_ID" 2>/dev/null && exit 0
if [ -n "$RUNPOD_API_KEY" ]; then
  curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" && exit 0
fi
echo "Could not stop the pod automatically. Please stop it on the RunPod website."
exit 1
