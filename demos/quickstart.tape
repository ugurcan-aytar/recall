# quickstart.tape — what an installed user sees in their first 30 seconds.
# Render with: make demos          (or: vhs demos/quickstart.tape)
# Output → assets/quickstart.gif

Output assets/quickstart.gif

Set Theme "Dracula"
Set FontSize 16
Set Width 1200
Set Height 760
Set Padding 20
Set TypingSpeed 50ms
Set Shell zsh

# Hidden setup: clean DB, put the locally-built `recall` on PATH so the
# typed commands look exactly like what a brew-installed user would type.
Hide
Type "cd /Users/ugurcanaytar/Desktop/Projects/Recall/recall && rm -rf /tmp/recall-demo && mkdir -p /tmp/recall-demo && export RECALL_DB_PATH=/tmp/recall-demo/index.db && export PATH=$PWD/demos:$PATH && setopt interactive_comments && export PROMPT='$ ' && clear" Enter
Sleep 600ms
Show

Type "# point recall at a folder of notes (the bundled examples here)" Enter
Sleep 800ms
Type "recall collection add ./examples --name notes" Enter
Sleep 1500ms

Type "recall index" Enter
Sleep 1500ms

Type "# full-text search — instant, no model needed" Enter
Sleep 800ms
Type "recall search 'circuit breaker'" Enter
Sleep 3500ms

Type "# retrieve a single document, first 8 lines" Enter
Sleep 800ms
Type "recall get notes/technical-doc.md --lines 8" Enter
Sleep 3500ms

Type "# what's installed and what's indexed" Enter
Sleep 800ms
Type "recall doctor" Enter
Sleep 2500ms

Type "recall status" Enter
Sleep 3500ms
