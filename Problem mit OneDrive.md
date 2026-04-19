# Problembehandlung: OneDrive & Virtuelle Umgebungen (.venv)

Virtuelle Python-Umgebungen (`.venv`) und Cloud-Speicher wie **OneDrive** führen oft zu Konflikten. Dieses Dokument erklärt, warum das passiert und wie man es löst.

## 🚩 Das Problem
OneDrive versucht, jede Datei in Echtzeit zu synchronisieren. Eine `.venv` enthält jedoch tausende winzige Dateien. Dies führt zu:
1. **Dateisperren**: OneDrive blockiert Dateien während des Syncs, sodass Python sie nicht lesen/schreiben kann.
2. **Pfad-Fehler**: "Fatal error in launcher". Wenn der Projektordner verschoben oder umbenannt wird (z.B. Rebranding), brechen die absoluten Pfade innerhalb der `.venv` Executables.
3. **Performance**: Der massive Sync-Prozess verlangsamt das gesamte System.

## 🛠️ Die Lösung: Reset der Umgebung
Wenn die App nicht mehr startet, ist der schnellste Weg ein "Clean Install" der Umgebung. Führe diese Befehle im Projektordner aus:

```powershell
# 1. Alte Umgebung löschen
Remove-Item -Recurse -Force .venv

# 2. Neue Umgebung erstellen
python -m venv .venv

# 3. Pakete neu installieren
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 🛡️ Vorbeugung
Um das Problem dauerhaft zu minimieren:
* **Pfad-Stabilität**: Benenne den Projektordner nicht um, während eine `.venv` darin aktiv ist.
* **Sync deaktivieren**: Rechtsklick auf den Ordner `.venv` -> *Einstellungen* -> *Diesen Ordner nicht synchronisieren* (falls möglich).
* **Lokale Kopie**: Entwickle Projekte idealerweise in einem lokalen Ordner (z.B. `C:\Projekte\`) und nutze **GitHub** für das Backup statt OneDrive. GitHub ignoriert die `.venv` bereits automatisch (via `.gitignore`).
