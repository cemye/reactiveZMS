# reactiveZMS

Übersicht über die Datein:
- dynaslot.py: ursprüngliches System (mit kleinen Anpassungen um die Ausführung zu ermöglichen)
- reactiveZMS.py: ursprüngliches System + Erweiterung (vollstädinge dokumentiert inklusive Änderungen )
- predictive Testbench: zum Testen der prädiktiven Ablaufplanung
- reactive Testbench: zum Testen der reaktiven Optimierung

- Lagerdaten: Liste mit Aufträgen und Rampen
  - lager_klein.json
  - lager_mittel.json
  - lager_gross.json
- ereignisse.json: Liste mit Ereignissen

Zum Ausführen des Programms wird Python benötigt.
In den Systemen werden diverse globale Parameter definiert:

- **DATASET**: Wähle den Datensatz aus: "lager_gross", "lager_mittel", "lager_klein" (*String*)
- **PATH**: Wo sind Datein gespeichert? - Dateipfad
- **methodePlan**: Wähle Bewertungsfunktion für Gesamtplan: "sum of tardiness", "mean of tardiness", "tardy orders", "sum of quad tardiness" - (*String*)
- **scoreThreshold**: Wähle Grenze für Neuplanung: 1 ist gleichbleibende Qualität, Verschlechterung bedeutet scoreThreshold > 1 (*Integer*)
- **show_plots**: Sollen Diagramme in der IDE angezeigt werden? - (*Boolean*)
- **generate_opt_plots**: Sollen Diagramme bei der Optimierung generiert werden? - (*Boolean*)
- **save_plots**: Sollen Diagramme gespeichert werden? - (*Boolean*)
- **weight**: Welches Gewicht soll für den Erhaltungsfakore der Bewertungsfunktion genutzt werden (*Integer*)
- **runs**: Wie viele Durchläufe sollen bei den Testläufen durchgeführt werden? (*Integer*)
- **event**: Soll ein Ereignis nach der prädiktiven Optimierung durchgeführt werden? - Wert orientiert sich an der Ereignistabelle (*None*/*Integer*)
