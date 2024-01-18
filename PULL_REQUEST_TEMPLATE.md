# Watson+Holmes PR Template

Dankjewel voor de PR! Gelieve de onderstaande checklist te controleren:

## 1. Documentatie (op git)
   - [ ] Duidelijk doel van de query
     - [ ] Doel van het proces
     - [ ] Uiteindelijk doel
     - [ ] Meerwaarde ten opzichte van bestaande tabellen
   - [ ] Verwijzing naar achterliggende processen
   - [ ] Toelichting van arbitraire waarden
   - [ ] Inline comments
   - [ ] Toelichting van kolommen
     - [ ] De kolom 'jaar' is vrij duidelijk, dus behoeft geen uitleg
     - [ ] De kolom 'procentuele daling' behoeft dat wel. Wat is de basis van wanneer die gedaald is?
     - [ ] De kolom 'HuurwaardeM2' behoeft uitleg. Hoe is die tot stand gekomen? Komt het uit het AVM, gebaseerd op transacties, of beiden? Is het per maand, per jaar, per M2 per jaar?

## 2. Algemeen
   - [ ] Code zonder syntax errors
   - [ ] Begrijpbaarheid voor de lezer
   - [ ] Specifiek doel van de code
   - [ ] Toegevoegde bestanden
   - [ ] Correctheid van de code
   - [ ] Toelichting van complexere gedeeltes
   - [ ] Runt de code binnen tijdsnormen?
    - [ ] Gebruik Queryplan of profilers om te analyseren

## 3. Code
   - [ ] Efficiënt gebruik van loops
     - [ ] Loops lopen niet vaker dan nodig
     - [ ] Loops in loops worden zoveel mogelijk vermeden
   - [ ] Efficiënt gebruik van joins
     - [ ] Joins vinden plaats op een zo klein mogelijke tabel
     - [ ] Joins sluiten geen data uit (Dit is duidelijk toegelicht)
   - [ ] Expliciete logica
     - [ ] Case when / if statements zijn expliciet en duidelijk
   - [ ] Juiste volgorde van filters
     - [ ] Filters die het meest filteren staan vooraan
   - [ ] Geen onnodige comment-code
   - [ ] DRY-principe toegepast
     - [ ] Gebruik functies en CTEs
   - [ ] Modulariteit van de code
     - [ ] Is het duidelijk dat elk stuk van de code zijn eigen doel heeft met concrete in en outputs?

## 4. SQL specifiek
   - [ ] Gebruik van SQL Formatter
     - [ ] [Poor SQL Formatter](https://poorsql.com/)
   - [ ] Juiste kolomnamen
     - [ ] [SQL Database Templates](https://git.watsonholmes.nl/WatsonHolmes/SqlDatabase/src/branch/main/General/Templates)
   - [ ] Juiste kolom volgorde. Eerst keys, (eventuele) belangrijke kolommen, alfabetisch, modifydate
   - [ ] Correct create statement
     - [ ] Primary key zinvol en integer?
     - [ ] Foreign keys aanwezig indien nodig?
   - [ ] Focus op runtijd voor views
     - [ ] Extra focus op runtijd


## 5. Python specifiek
   - [ ] PEP 8-naleving (flake8 / ruff)
   - [ ] Aanwezigheid van docstrings
   - [ ] Gesorteerde imports (isort)
   - [ ] Robuustheid met tests
    - [ ] Unit tests
   - [ ] Consistentie in variabele namen
     - [ ] Lowercase
     - [ ] Variabele die niet van naam veranderen: UPPERCASE
     - [ ] Classes: capitalized
   - [ ] Adequate error handling (Geen bare except)
   - [ ] Gedefinieerde en relatieve paden

## 6. Overig
    - [ ] Extra nadruk op uitlegbare code

## 7. Duidelijke beschrijving na de PR
   - [ ] Moet de query gescheduled worden? Zoja, wanneer en met welke frequentie?
   - [ ] Eenmalig runnen op SQL02?
