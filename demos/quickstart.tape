# quickstart.tape — recall in 30 seconds: index sample notes, run BM25.
# Render with: vhs demos/quickstart.tape   (output → assets/quickstart.gif)

Output assets/quickstart.gif

Set Theme "Dracula"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 20
Set TypingSpeed 50ms
Set Shell zsh

# Sandbox + zsh prep so `# narration` lines work as comments and the prompt
# stays minimal during the recording.
Hide
Type "cd /Users/ugurcanaytar/Desktop/Projects/Recall/recall && rm -rf /tmp/recall-demo && mkdir -p /tmp/recall-demo && export RECALL_DB_PATH=/tmp/recall-demo/index.db && setopt interactive_comments && export PROMPT='$ ' && clear" Enter
Sleep 600ms
Show

Type "# point recall at the bundled examples (meeting notes, runbook, journal)" Enter
Sleep 800ms
Type "demos/recall collection add ./examples --name notes" Enter
Sleep 1500ms

Type "demos/recall index" Enter
Sleep 1500ms

Type "# full-text search — instant, no model needed" Enter
Sleep 800ms
Type "demos/recall search 'circuit breaker'" Enter
Sleep 3500ms

Type "# retrieve a single document by path" Enter
Sleep 800ms
Type "demos/recall get notes/technical-doc.md --lines 8" Enter
Sleep 3500ms

Type "demos/recall status" Enter
Sleep 3500ms
