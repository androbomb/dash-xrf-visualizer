# Messaggio iniziale per Hermes (copia il testo del blocco)

```text
Ciao Hermes. Avvia l'audit della mia vecchia applicazione Plotly Dash per dati XRF.

PROJECT_ROOT=/workspace/exchange/dash-xrf-visualizer
Verifica che la directory esista sul TUO filesystem; se manca, fermati. È la repo del codice, non il path dei dati XRF, che sarà una variabile Python nel notebook.

OBIETTIVO FINALE (NON implementare in questo primo passaggio):
1. Rifattorizzare e semplificare il progetto. Mantieni SOLO il visualizzatore: file XRF HDF5 (aggiungere .npy/.npz), immagine 2-D a SINISTRA e spettro a DESTRA. Selezione intervallo spettrale → ricalcolo della mappa mediante somma/integrazione; selezione ROI nell'immagine → ricalcolo dello spettro. Verifica calibrazione/assi/unità, differenza zoom vs ROI e regole di aggregazione.
2. ELIMINARE la parte AIRES, modello, inferenza, training e UI collegata (decisione già presa). Eventuale lettore custom: rimandare; segnalalo, non migrare ora.
3. Avvio da Jupyter/JupyterLab con `path_to_file = "/path/to/file"` e factory Dash; nessun path hardcoded. Calcolo JAX dove ha senso, con rilevamento del backend GPU effettivamente supportato e fallback CPU esplicito; NumPy/h5py consentiti per I/O e interoperabilità. Preserva la correttezza XRF.
4. Proporre una repo leggibile, root essenziale (Markdown, manifest/lock uv e UN notebook di esempio): `src/xrf_visualizer/` con moduli separati data/processing/ui (layout HTML/Dash, callback, figure), `tests/` per pytest, `deploy/docker/` (Dockerfile/Compose), `deploy/wslc/` (Containerfile), `deploy/apptainer/` (.def), `deploy/uv/` (istruzioni venv e kernel Jupyter). `pyproject.toml` e `uv.lock` nella ROOT; `.venv` ignorata. Puoi usare uv dopo approvazione, NON durante M0. Non potrai testare Docker/WSLC/Apptainer: prepara in futuro ricette e documentazione marcate NON TESTATE; le proverò io.
5. `dash-xrf-analyzer/` è SOLO un riferimento legacy di bootstrap: va gitignorata e NON deve essere importata o usata dal nuovo runtime. Puoi esaminarla in sola lettura; NON eliminarla, decideremo più avanti.

ADESSO SOLO M0: AUDIT IN LETTURA.
- Leggi `AGENTS.md`, `operational_plan.md`, `to_do.md`, `timeline.md`, se presenti. Il README marcato BOOTSTRAP_TEMPLATE_README è provvisorio.
- Controlla `git status`, architettura, entrypoint, layout, callback, pipeline del cubo XRF, reader, AIRES, dipendenze e test. Per il file XRF di esempio esamina SOLTANTO metadati e piccoli campioni; niente caricamenti massivi.
- Restituisci tabella: funzionalità, file/funzioni, dipendenze, KEEP / REMOVE / DEFER / DA CHIARIRE, motivazione. Non richiedere nuovamente se conservare AIRES o le funzioni del visualizzatore: le decisioni sono sopra.
- Proponi il tree minimo, le migrazioni, i test di parità numerica e CPU/GPU, e una milestone M1 piccola. Distingui osservazioni da ipotesi; indica le modifiche ai Markdown ma NON applicarle.

VINCOLI M0: nessuna scrittura/modifica/rimozione; nessuna installazione, download, server, test oneroso, commit o push. Produci il rapporto con rischi e domande tecniche strettamente necessarie, poi FERMATI in attesa della mia approvazione.
```
