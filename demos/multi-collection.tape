# multi-collection.tape — searching across two repos at once with -c repo1,repo2
# Render with: vhs demos/multi-collection.tape   (output → assets/multi-collection.gif)

Output assets/multi-collection.gif

Set Theme "Dracula"
Set FontSize 16
Set Width 1200
Set Height 720
Set Padding 20
Set TypingSpeed 50ms
Set Shell zsh

Hide
Type "cd /Users/ugurcanaytar/Desktop/Projects/Recall/recall && rm -rf /tmp/recall-multi && mkdir -p /tmp/recall-multi && export RECALL_DB_PATH=/tmp/recall-multi/index.db && setopt interactive_comments && export PROMPT='$ ' && demos/recall collection add ./examples --name notes >/dev/null && demos/recall collection add ./internal --name src --mask '**/*.go' >/dev/null && demos/recall index >/dev/null && clear" Enter
Sleep 800ms
Show

Type "# two collections indexed: hand-written notes + recall's own Go source" Enter
Sleep 800ms
Type "demos/recall collection list" Enter
Sleep 3500ms

Type "# searching only the notes collection" Enter
Sleep 800ms
Type "demos/recall search 'rate limiter' -c notes -n 3" Enter
Sleep 3500ms

Type "# searching only the source collection" Enter
Sleep 800ms
Type "demos/recall search 'fusion' -c src -n 3" Enter
Sleep 3500ms

Type "# both collections at once with -c notes,src" Enter
Sleep 800ms
Type "demos/recall search 'authentication' -c notes,src -n 4" Enter
Sleep 4000ms
