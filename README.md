# Analisi del surplus/deficit energetico in Europa
## 0.1 Introduzione
---
Il seguente progetto ha lo scopo di analizzare il deficit e il surplus di energia elettrica egli stati membri dell'Unione Europea nel tempo, concentrandosi sulla produzione e sul consumo locale, escludendo così fattori esterni come importazioni ed esportazioni di elettricità.
I database in analisi sono i seguenti:
- Produzione di energia: https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_pem/default/table?lang=en&category=nrg.nrg_quant.nrg_quantm.nrg_cb_m
- Consumo di energia: https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_e/default/table?lang=en&category=nrg.nrg_quant.nrg_quanta.nrg_cb
## 0.2 Come far partire la webapp
---
### Parte 1: scarica il file
#### Opzione 1: Scaricalo come file zip
1. Clicca sul pulsante verde `<> Code`  posizionato in alto a destra
![PTasto <> Code](https://github.com/SteTerro/Progetto/blob/main/tutorial1.png?raw=true|100)
2. Clicca su `Download ZIP`
3. Scegli la directory in cui salvare il file
4. Trova il file, unzippalo usando una tua app a piacimento (7-zip, WinRAR, etc..) 
#### Alternativa: scaricalo da git
1. Scegli la cartella in cui vuoi scaricare il progetto
2. Apri il terminale del tuo dispositivo (PowerShell, terminale macOS, terminale Linux, etc...)
3. Spostati sulla cartella scelta in precedenza
   - Per Windows: `cd C:\Users\YourUser\\...`
   - Per Linux\macOS: `cd /Users/YourUser/...`
(Con `YourUser` viene indicato il nome utente del pc, e `...` il path di sistema da completare)
5. Copiare il comando: `git@github.com:SteTerro/Progetto.git`
6. Trova il file, unzippalo usando una tua app a piacimento (7-zip, WinRAR, etc..) 

### Parte 2: esegui il codice
#### Opzione 1: Da un IDE
1. Apri il tuo IDE di scelta (es. Visual Studio Code)
2. Clicca su `File` in alto a sinistra
3. Clicca su `Open Folder...`
(In alternativa ai passaggi 2. e 3. si può usare la combinazione `CTRL + K + O`)
4. Seleziona la **cartella** in cui è stato precedentemente scaricato il file
5. Una volta che tutto si è aperto, digita sul terminale aperto in basso:
		`uv run streamlit run webapp.py`   
6. Clicca Invio

#### Alternativa: esegui dal terminale
1. Apri il terminale (o ritornaci se è ancora aperto)
2. Spostati nella cartella del progetto appena scaricato
	- Per Windows: `cd C:\Users\YourUser\\...`
	- Per Linux\macOS: `cd /Users/YourUser/...`
(Se si è scaricato la repository tramite terminale bisogna comunque spostarsi di nuovo)
3. Una volta che si è nella cartella giusta, digitare nel terminale: 
		`uv run streamlit run webapp.py`
4. Cliccare invio
 
> Se ci sono problemi nell'eseguire l'app verificare ancora una volta che si è nella cartella giusta!

## Analisi
---
Se si vuole capire meglio come sono state svolte le analisi e il perchè dietro alcune scelte è possibile farlo leggendo la [documentazione allegata](https://github.com/SteTerro/Progetto/blob/main/Documentazione.pdf)
