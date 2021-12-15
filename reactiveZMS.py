"""
Für die einzelnen Ereignisse werden mithilfe vom Pandas Framework Tabellen erstellt und mit den übergebenen Daten befüllt.
Außerdem werden die Daten für die Zeitfensteranfragen in eine seperate Tabelle abgespeichert.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from datetime import datetime as dt
import json
import pandas as pd
import random

"""---------------------------------------Methoden---------------------------------------"""
"""
In diesem Abschnitt werden die Methoden definiert, welche im Ablauf genutzt werden.
Es sind folgende Kennzeichnungen im Code zu finden: "Neue Methode" "Erweiterte Methode"
"""


"""Eine Funktion die aus einem Zeit-String die genaue Sekunden Anzahl berechnet. Beispielsweise wird eine Stunde zu 3600 Sekunden verrechnet. Diese Funktion wird für die Berechnung der **duration** benötigt"""


def get_sec(time_str):
    h, m = time_str.split(':')
    return int(h) * 3600 + int(m) * 60


"""Hier werden die zuvor eingelesenen Zeitfensteranfragen verarbeitet und in die Zeitfenster-Tabelle abgespeichert. **Timestamp** und **duration** werden zu Sekunden umgerechnet."""


def preprocess(orders, number_of_ramps):
    orders = orders.copy(deep=True)

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    orders["timestamp"] = orders["timestamp"].map(lambda t: t.timestamp())
    orders["desired_timestamp"] = orders["timestamp"]
    orders["duration"] = orders["duration"].map(lambda t: get_sec(t))
    orders["endtimes"] = orders["timestamp"] + orders["duration"]
    orders["ramp"] = np.random.randint(1, number_of_ramps, size=len(orders))
    orders["slot_id"] = np.arange(len(orders.index))

    return orders


"""Im Folgenden werden die Ereignisse einzeln verarbeitet, sodass sie später im Programm genutzt werden können."""


def preprocess_added_ramps(added_ramps):
    added_ramps = added_ramps.copy(deep=True)

    added_ramps["timestamp"] = pd.to_datetime(added_ramps["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    added_ramps["timestamp"] = added_ramps["timestamp"].map(lambda t: t.timestamp())

    return added_ramps


def preprocess_deleted_ramps(deleted_ramps):
    deleted_ramps = deleted_ramps.copy(deep=True)

    deleted_ramps["timestamp"] = pd.to_datetime(deleted_ramps["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    deleted_ramps["timestamp"] = deleted_ramps["timestamp"].map(lambda t: t.timestamp())

    return deleted_ramps


def preprocess_added_slots(added_slots):
    added_slots = added_slots.copy(deep=True)

    added_slots["timestamp"] = pd.to_datetime(added_slots["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    added_slots["timestamp"] = added_slots["timestamp"].map(lambda t: t.timestamp())
    added_slots["timestamp_slot"] = pd.to_datetime(added_slots["timestamp_slot"], format="%Y-%m-%d_%H:%M:%S")
    added_slots["timestamp_slot"] = added_slots["timestamp_slot"].map(lambda t: t.timestamp())
    added_slots["duration"] = added_slots["duration"].map(lambda t: get_sec(t))
    added_slots["endtimes"] = added_slots["timestamp_slot"] + added_slots["duration"]

    return added_slots


def preprocess_deleted_slots(deleted_slots):
    deleted_slots = deleted_slots.copy(deep=True)

    deleted_slots["timestamp"] = pd.to_datetime(deleted_slots["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    deleted_slots["timestamp"] = deleted_slots["timestamp"].map(lambda t: t.timestamp())

    return deleted_slots


def preprocess_delayed_slots(delayed_slots):
    delayed_slots = delayed_slots.copy(deep=True)

    delayed_slots["timestamp"] = pd.to_datetime(delayed_slots["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    delayed_slots["timestamp"] = delayed_slots["timestamp"].map(lambda t: t.timestamp())
    delayed_slots["delay"] = delayed_slots["delay"].map(lambda t: get_sec(t))

    return delayed_slots


def preprocess_maintainance(maintainance):
    maintainance = maintainance.copy(deep=True)

    maintainance["timestamp"] = pd.to_datetime(maintainance["timestamp"], format="%Y-%m-%d_%H:%M:%S")
    maintainance["timestamp"] = maintainance["timestamp"].map(lambda t: t.timestamp())
    maintainance["timestamp_slot"] = pd.to_datetime(maintainance["timestamp_slot"], format="%Y-%m-%d_%H:%M:%S")
    maintainance["timestamp_slot"] = maintainance["timestamp_slot"].map(lambda t: t.timestamp())
    maintainance["duration"] = maintainance["duration"].map(lambda t: get_sec(t))
    maintainance["endtimes"] = maintainance["timestamp_slot"] + maintainance["duration"]

    return maintainance


"""Zwei Funktionen für die Planerstellung.
*max_overlap_in_ramp* überprüft für die übergebene Rampe,
ob es für das gewünschte Zeitfenster Überschneidungen auf der Rampe gibt oder ob die Rampe vielleicht schon gelöscht wurde.
Für die Überprüfung von Überschneidungen wird die Funktion *slot_overlap* aufgerufen, 
um jedes bereits verbuchte Zeitfenster mit dem neuen zu vergleichen und zu berechnen, wie stark sie sich überschneiden. 
Eine Liste mit den Überschneidungen auf der Rampe wird zurück an die aufrufende Funktion gegeben."""


def slot_overlap(slot1, slot2):
    if slot1["slot_id"] == slot2["slot_id"]:
        return -600
    return min(slot1["endtimes"], slot2["endtimes"]) - max(slot1["timestamp"], slot2["timestamp"])


def max_overlap_in_ramp(order, schedule, ramp):
    if ramp in deleted_ramps:
        return np.inf

    slots_for_ramp = schedule.loc[schedule["ramp"] == ramp]
    if slots_for_ramp.empty:
        return -600

    overlaps = [slot_overlap(slot, order) for _, slot in slots_for_ramp.iterrows()]
    return max(overlaps)


"""*Delay_fit* ist für die Erstellung des Plans zuständig.
In dieser Funktion werden alle Anfragen durchlaufen, hier **orders** genannt.
Dann wird die zuvor genannte Funktion *max_overlaps_in_ramp* für jede Rampe einmal aufgerufen, 
um eine Liste der Überschneidungen auf allen Rampen zu erhalten.
Dann wird die Rampe bestimmt bei *best_ramp*,
wo sich das Wunschzeitfenster und die bereits verplanten Zeitfenster am wenigsten überschneiden.
Auf der besten Rampe wird dann das Zeitfenster solange verschoben, bis es eine freie Stelle im Plan gibt."""


def delay_fit(orders, number_of_ramps):
    schedule = orders.iloc[0:0].copy(deep=True)
    for i in range(len(orders)):
        order = orders.iloc[i]

        ramp_overlaps = [max_overlap_in_ramp(order, schedule, ramp) for ramp in range(number_of_ramps)]
        best_ramp = np.random.choice(np.argwhere(np.array(ramp_overlaps) == np.array(ramp_overlaps).min())[0])

        order["ramp"] = best_ramp

        while max_overlap_in_ramp(order, schedule, best_ramp) > 0:
            overlap = max_overlap_in_ramp(order, schedule, best_ramp)
            order["timestamp"] += overlap + 600
            order["endtimes"] = order["timestamp"] + order["duration"]

        schedule = schedule.append(order)
        yield schedule


"""Überprüft ob Zeitfenster sich im aktuellen Plan überschneiden. Gibt einen Boolean zurück."""


def is_overlapping(solution):
    for ramp in solution["ramp"].unique():
        slots = solution.loc[solution["ramp"] == ramp]
        sorted = slots.sort_values("timestamp")
        for i in range(len(sorted.index) - 1):
            if slot_overlap(sorted.iloc[i], sorted.iloc[i + 1]) > 0:
                return True
    return False


"""
Enkodiert den Zeitplan, sodass schnell neue Lösungen generiert werden können.
In *decode* werden die erstellten Lösungen wieder in Tabellenform gebracht.
"""


def encode(schedule):
    encoded = [[slot["timestamp"], slot["ramp"]] for _, slot in schedule.iterrows()]
    return np.array(encoded)


"""
Erweiterte Methode: 
Nach Löschen eines Auftrages wird ebenfalls die Referenz im Schedule gelöscht. 
Daher haben encoded und schedule zwar die selbe Länge, aber da eine Referenz im schedule gelöscht wurde, sind die stimmen die Referenzen nicht überein.
Beispiel: Auftrag 2 wurde gelöscht:
schedule = [0, 1, 3, 4, 5]
encoded = [0, 1, 2, 3, 4]
"""

def decode(encoded, schedule):
    new_schedule = schedule.copy(deep=True)

    missing_index = False
    for i in range(len(encoded)):
        if i in new_schedule.index and missing_index is False:
            new_schedule.loc[i, "timestamp"] = encoded[i][0]
            new_schedule.loc[i, "endtimes"] = encoded[i][0] + new_schedule.loc[i, "duration"]
            new_schedule.loc[i, "ramp"] = encoded[i][1]
        else: # sorgt für das Überspringen der gelöschten Referenz und das Angleichen der Referenzen
            missing_index = True
            new_schedule.loc[i + 1, "timestamp"] = encoded[i][0]
            new_schedule.loc[i + 1, "endtimes"] = encoded[i][0] + new_schedule.loc[i + 1, "duration"]
            new_schedule.loc[i + 1, "ramp"] = encoded[i][1]

    return new_schedule


"""
Funktion zur Darstellung der Gantt Diagramme. Hier werden auch die Farben der Transportfahrzeuge festgesetzt und abgespeichert.
Erweiterte Methode: 
Es wurden graphische Steuerbefehle für die Darstellung in Colab entfernt.
"""


def plot(schedule, number_of_ramps):
    global plot_number, fig, gnt

    np.random.seed(42)
    ids = orders["id"].unique()
    colors_for_ids = [np.random.rand(3) for id in ids]

    fig, gnt = plt.subplots(dpi=200)
    fig.subplots_adjust(bottom=0.25)

    gnt.set_xlabel('Zeit')
    gnt.set_ylabel('Rampen')

    gnt.set_yticks([int(15 + x * 10) for x in range(number_of_ramps)])
    gnt.set_yticklabels([str(x) for x in range(number_of_ramps)])

    plt.xticks(rotation=45)

    gnt.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: dt.fromtimestamp(x).strftime("%d.%m %H Uhr")))

    gnt.grid(True)

    for ramp in range(number_of_ramps):
        slots = schedule.loc[schedule["ramp"] == ramp]
        slots["color"] = slots["id"].map(lambda id: colors_for_ids[int(id)])
        gnt.broken_barh(slots[["timestamp", "duration"]].to_numpy(), (ramp * 10 + 10, 9),
                        facecolors=slots["color"].to_numpy())

    return fig, gnt


"""
Neue Methode:
Je nach Konfiguration werden Diagramme in der IDE abngezeigt und/oder gespeichert.
"""
def showPlot():
    global plot_number, fig, save_plots, show_plots
    if save_plots:
        plt.savefig("plots\plot" + str(plot_number) + ".png")
        plot_number += 1
    show_plots and fig.show()
    plt.close(fig)


"""Die Wahrscheinlichkeit, dass eine schlechtere Lösung als die aktuelle Lösung im Simulated Annealing Algorithmus gewählt wird."""


def probability(temp, diff):
    p = np.exp(-(diff / temp))
    return p


"""
Verändert den Plan durch das Anpassen von Rampen (zeitliche Verschiebung bzw. Verschiebung der Rampen).
Erweiterte Methode:
Es wird eine neue Rampennummer generiert, wenn ein Auftrag auf eine gelöschte Rampe geplant werden soll.
"""


def mutation(x):
    x = np.copy(x)
    r = random.randint(0, len(x) - 1)
    if random.random() > 0.1:
        x[r][0] += random.gauss(0, 36000)
    else:
        number = random.randint(0, problem["number_ramps"] - 1)
        while number in list_of_deleted_ramps:
            number = random.randint(0, problem["number_ramps"] - 1)
        x[r][1] = number
    return x


"""Simulated Annealing Algorithmus. Temperatur, stopcondition und alpha werden an die Funktion übergeben."""


def simulated_annealing(f, stopcondition, x, temperature, alpha):
    iteration = 0
    while not stopcondition(iteration):
        iteration += 1
        x_new = mutation(x)
        diff = np.abs(f(x_new) - f(x))
        p = probability(temperature, diff)

        # print("Sim:")
        # print(f(x_new))
        # print(f(x))
        if f(x_new) <= f(x):
            x = x_new
            yield x, temperature, f(x)
        else:
            if random.random() < p:
                x = x_new
                yield x, temperature, f(x)

        temperature *= alpha


"""Die Zeitfenster werden bestimmt, die mit dem neu einzuplanenden Zeitfenster im Konflikt stehen."""


def overlap_ids_in_ramp(order, schedule, ramp):
    slots_for_ramp = schedule.loc[schedule["ramp"] == ramp]
    overlaps = np.array([slot_overlap(slot, order) for _, slot in slots_for_ramp.iterrows()])
    overlap_ids = np.arange(len(overlaps))[overlaps >= -600]
    overlap_ids = map(lambda id: slots_for_ramp.iloc[id]["slot_id"], overlap_ids)
    return list(overlap_ids)


"""
Die Planungsheuristik:
Hier wird wenn möglich eine freie Rampe zum gewünschten Zeitpunkt gesucht.
Falls alle Rampen belegt wird, werden die Fitnessfunktionsbewertungen verglichen.
Dabei wird die Bewertung des neu einzuplanenden Zeitfensters **x** mit der Summe aller Zeitfenster **y** verglichen, die mit **x** überschneiden.
Wenn **x** eine niedrigere Bewertung hat als **y**, wird es verschoben, ansonst müssen die Zeitfenster **y** verschoben werden.
"""


def compare_slot(schedule, order):
    ramp_overlaps = [max_overlap_in_ramp(order, schedule, ramp) for ramp in range(problem["number_ramps"])]

    # Wenn eine Rampe zum gewünschten Zeitpunkt frei ist, return die Rampe und das Zeitfenster
    if min(ramp_overlaps) <= -600:
        best_ramp = np.random.choice(np.argwhere(np.array(ramp_overlaps) == np.array(ramp_overlaps).min())[0])
        return best_ramp, order

    # Sonst Berechne die Zeitfenster die sich mit dem zu verbuchenden Fenster überschneiden und bestimme,
    # welche Bewertungen die Zeitfenster von der Zielfunktion bekommen.
    else:
        overlap_ids = overlap_ids_in_ramp(order, schedule, order["ramp"])
        score_overlaps = sum(
            [schedule.loc[schedule["slot_id"] == overlap_id]["score"].item() for overlap_id in overlap_ids])

        # Falls die Summe der bereits verplanten Zeitfenster größer ist als die Bewertung des
        # neuen Zeitfensters, wird das neue Zeitfenster verschoben
        if score_overlaps > order["score"]:
            best_ramp = np.random.choice(np.argwhere(np.array(ramp_overlaps) == np.array(ramp_overlaps).min())[0])
            while max_overlap_in_ramp(order, schedule, best_ramp) > 0:
                overlap = max_overlap_in_ramp(order, schedule, best_ramp)
                order["timestamp"] += overlap + 600
                order["endtimes"] = order["timestamp"] + order["duration"]

            return best_ramp, order

        # Ansonsten werden die bereits verplanten Zeitfenster verschoben und
        # das neu hinzugekommene Zeitfenster bekommt seinen gewünschten Zeitraum.
        else:
            for i, overlap_id in enumerate(overlap_ids):

                new_slot = schedule[schedule["slot_id"] == overlap_id].iloc[0]
                ramp_overlaps = [max_overlap_in_ramp(new_slot, schedule, ramp) for ramp in
                                 range(problem["number_ramps"])]
                best_ramp = np.random.choice(np.argwhere(np.array(ramp_overlaps) == np.array(ramp_overlaps).min())[0])
                while max_overlap_in_ramp(new_slot, schedule, best_ramp) > 0:
                    overlap = max_overlap_in_ramp(new_slot, schedule, best_ramp)
                    new_slot["timestamp"] += overlap + 600
                    new_slot["endtimes"] = new_slot["timestamp"] + new_slot["duration"]

                schedule.iloc[overlap_id.astype(int)] = new_slot

            return best_ramp, order


"""
Bei Ereignis *add_ramp* wird eine neue Rampe hinzugefügt.
Eine Anzahl von Aufträgen werden dabei erneut auf die bestmögliche Rampe verbucht, wozu nun auch die neue Rampe gehören kann.
Die Anzahl berechnet sich aus der Anzahl der Aufträge durch die Anzahl der Rampen.
Für die Umplanung wird die zuvor vorgestellte Planungsheuristik genutz.
"""


def add_ramp(decoded):
    problem["number_ramps"] += 1
    schedule = decoded.copy(deep=True)
    schedule.duration = schedule.duration.astype(int)
    schedule.ramp = schedule.ramp.astype(int)

    for i in range(int(len(schedule.index) / problem["number_ramps"])):
        calc_scores(schedule)

        index_order = schedule["score"].argmax()

        order = schedule.iloc[index_order]

        order["timestamp"] = order["desired_timestamp"]
        order["endtimes"] = order["desired_timestamp"] + order["duration"]

        best_ramp, order = compare_slot(schedule, order)
        order["ramp"] = best_ramp

        new_score = np.abs(order["timestamp"] - order["desired_timestamp"])
        if new_score < schedule.at[index_order, "score"]:
            schedule.iloc[index_order] = order
        else:
            break

    return schedule


"""
*Maintenance* erstellt ein Zeitfenster, was alle anderen Fenster in diesem Zeitraum verschiebt.
Maintenance Instanzen besitzen immer die ID -1 und werden nur durch das Löschen der Rampe entfernt.
"""


def maintenance(start_time, end_time, maintained_ramp, new_schedule):
    calc_scores(new_schedule)
    new_schedule.loc[len(new_schedule.index)] = [-1, start_time, end_time - start_time, start_time, end_time,
                                                 maintained_ramp, len(new_schedule.index), 0]
    new_schedule.astype(float)

    def switch_ramp(order):
        current_ramp = order["ramp"]
        if current_ramp == maintained_ramp:
            best_ramp, order = compare_slot(new_schedule, order)
            order["ramp"] = best_ramp

        return order

    for i in range(len(new_schedule.index)):
        if new_schedule.iloc[i]["timestamp"] > start_time and new_schedule.iloc[i]["timestamp"] < end_time:
            new_schedule.iloc[i] = switch_ramp(new_schedule.iloc[i])

        if new_schedule.iloc[i]["timestamp"] < start_time and new_schedule.iloc[i]["timestamp"] < end_time:
            new_schedule.iloc[i] = switch_ramp(new_schedule.iloc[i])

        if new_schedule.iloc[i]["timestamp"] > start_time and new_schedule.iloc[i]["timestamp"] > end_time:
            new_schedule.iloc[i] = switch_ramp(new_schedule.iloc[i])

    return new_schedule


"""
Bei dem Ereignis *delete_ramp* wird eine Rampe aus dem Plan entfernt.
Alle auf dieser Rampe sich befindenden Zeitfenster müssen neu verplant werden.
"""


def delete_ramp(deleted_ramp, new_schedule2):
    new_schedule = new_schedule2.copy(deep=True)
    calc_scores(new_schedule)

    # Funktion um Maintenance Instanzen zu löschen, es wird überprüft ob ein Zeitfenster die ID -1 besitzt.
    new_schedule = new_schedule.drop(
        new_schedule[(new_schedule["ramp"] == deleted_ramp) & (new_schedule["id"] == -1)].index)

    def switch_ramp(order):
        current_ramp = order["ramp"]
        if current_ramp == deleted_ramp:
            best_ramp, order = compare_slot(new_schedule, order)
            order["ramp"] = best_ramp
        return order

    deleted_ramps.append(deleted_ramp)
    list_of_deleted_ramps.append(deleted_ramp)
    for i in range(len(new_schedule.index)):
        new_schedule.iloc[i] = switch_ramp(new_schedule.iloc[i])

    return new_schedule


"""Das Ereignis Zeitfenster hinzufügen nutzt die Planungsheuristik, um das neue Zeitfenster zuzuweisen."""


def add_slot(new_schedule, start_time, duration, end_time, id):
    calc_scores(new_schedule)

    new_schedule.loc[len(new_schedule.index)] = [id, start_time, duration, start_time, end_time, 2,
                                                 len(new_schedule.index), 0]
    order = new_schedule.iloc[-1]
    best_ramp, order2 = compare_slot(new_schedule, order)
    order2["ramp"] = best_ramp

    return new_schedule


"""Das Ereignis *delete_slot* entfernt einfach das entsprechende Zeitfenster."""


def delete_slot(new_schedule, slot_id):
    return new_schedule.drop(new_schedule[new_schedule["slot_id"] == slot_id].index)


"""Delay Slot verschiebt ein bereits existierendes Zeitfenster und nutzt dann die Planungsheuristik für die neue Einordnung."""


def delay_slot(new_schedule, index, delay):
    calc_scores(new_schedule)
    slot = new_schedule.iloc[index]

    slot["timestamp"] += delay
    slot["endtimes"] = slot["timestamp"] + slot["duration"]

    best_ramp, slot = compare_slot(new_schedule, slot)
    slot["ramp"] = best_ramp
    return new_schedule


"""
Die *event_queue* soll die richtigen Funktionen basierend auf der *events* Tabelle ausführen.
Dafür bekommt sie sequentiell alle event types aus dieser Tabelle übergeben.
"""


def event_queue(new_schedule2, event_type):
    for type_added_ramps in range(len(preprocessed_added_ramps)):
        if events.iloc[event_type]["type"] == "add_ramp":
            new_schedule2 = add_ramp(new_schedule2)
            return new_schedule2

    for type_deleted_ramps in range(len(preprocessed_deleted_ramps)):
        if events.iloc[event_type]["type"] == "delete_ramp":
            new_schedule2 = delete_ramp(preprocessed_deleted_ramps.iloc[type_deleted_ramps]["ramp"], new_schedule2)

            return new_schedule2

    for type_added_slots in range(len(preprocessed_added_slots)):
        if events.iloc[event_type]["type"] == "add_slot":
            new_schedule2 = add_slot(new_schedule2, preprocessed_added_slots.iloc[type_added_slots]["timestamp_slot"],
                                     preprocessed_added_slots.iloc[type_added_slots]["duration"],
                                     preprocessed_added_slots.iloc[type_added_slots]["timestamp_slot"] +
                                     preprocessed_added_slots.iloc[type_added_slots]["duration"],
                                     preprocessed_added_slots.iloc[type_added_slots]["id"])
            return new_schedule2

    for type_deleted_slots in range(len(preprocessed_deleted_slots)):
        if events.iloc[event_type]["type"] == "delete_slot":
            new_schedule2 = delete_slot(new_schedule2, preprocessed_deleted_slots.iloc[type_deleted_slots]["id"])
            return new_schedule2

    for type_delayed_slots in range(len(preprocessed_delayed_slots)):
        if events.iloc[event_type]["type"] == "delay_slot":
            new_schedule2 = delay_slot(new_schedule2, preprocessed_delayed_slots.iloc[type_delayed_slots]["id"],
                                       preprocessed_delayed_slots.iloc[type_delayed_slots]["delay"])
            return new_schedule2

    for type_maintainance in range(len(preprocessed_maintainance)):
        if events.iloc[event_type]["type"] == "maintainance":
            new_schedule2 = maintenance(preprocessed_maintainance.iloc[type_maintainance]["timestamp_slot"],
                                        preprocessed_maintainance.iloc[type_maintainance]["timestamp_slot"] +
                                        preprocessed_maintainance.iloc[type_maintainance]["duration"],
                                        preprocessed_maintainance.iloc[type_maintainance]["ramp"], new_schedule2)
            return new_schedule2


"""
Optimierung des Plans mit der zuvor erwähnten Simulated Annealing Funktion.
Temperatur, stopcondition (Iterationen) und alpha können hier verändert werden.
Die Optimierung wird wieder als GIF und Tabelle verdeutlicht, wobei t die Temperatur und fitness den Fitnessfunktionswert abbildet.
Erweiterte Methode:
Es wurde eine boolean Option zur reaktiven Optimierung eingefügt.
Global kann eingestellt werden, ob Pläne während der Optimierung generiert werden sollen.
"""


def optimize(schedule, reactive):
    global fig, gnt, decoded, currentBestScore, fit, newest_schedule

    print("Optimizing...")
    encoded = encode(schedule)
    if generate_opt_plots:
        fig, gnt = plot(schedule, problem["number_ramps"])
        gnt.set_title(f"t = {np.round(200, 2)}, fitness = {np.round(fitness(encoded, schedule), 2)}")
        showPlot()
    oldschedule = schedule  # Alter Ablaufplan für den Erhaltungsfaktor der Bewertungsfunktion der reaktiven Optimierung.

    if reactive is True:  # reaktive Optimierung
        for result, temperature, fit in simulated_annealing(
                f=lambda x: reaction_fitness(x, schedule, oldschedule),  # nutzt reaktive Bewertungsfunktion
                stopcondition=lambda i: i > 900,
                x=encoded,
                temperature=200,
                alpha=0.99
        ):
            decoded = decode(result, schedule)
            if generate_opt_plots:
                fig, gnt = plot(decoded, problem["number_ramps"])
                gnt.set_title(f"t = {np.round(temperature, 2)}, fitness = {np.round(fit, 2)}")
                showPlot()
    else:  # prädiktive Optimierung
        for result, temperature, fit in simulated_annealing(
                f=lambda x: fitness(x, schedule), # nutzt prädiktive Bewertungsfunktion
                stopcondition=lambda i: i > 900,
                x=encoded,
                temperature=200,
                alpha=0.99
        ):
            decoded = decode(result, schedule)
            if generate_opt_plots:
                fig, gnt = plot(decoded, problem["number_ramps"])
                gnt.set_title(f"t = {np.round(temperature, 2)}, fitness = {np.round(fit, 2)}")
                showPlot()

    currentBestScore = fitness(encode(decoded), schedule)  # Globale Bewertung wird nach der Optimierung überschrieben
    print("Optimizing finished")
    return decoded


"""Berechnung der Fitnessfunktionsbewertung für jedes einzelne Zeitfenster."""


def calc_scores(schedule):  # todo
    schedule['score'] = (schedule['timestamp'] - schedule['desired_timestamp']).abs()


"""
Neue Methode:
Überprüft anhand des Bewertungsgrenze scoreThreshold, ob optimiert werden muss
"""


def need_optimization(score):
    return True if (scoreThreshold < (score / currentBestScore)) else False


"""
Berechnet die Differenz zwischen gewünschtem und erhaltenem Zeitfenster (Zielfunktion).
Überprüft auch, ob eine erstellte Lösung gültig ist durch den Aufruf von *is_overlapping*.
Erweiterte Methode: 
Dies wird für die prädiktive Optimierung genutzt.
Es stehen mehrere Bewertungsfunktionen zur Auswahl.
Die Auswahl wird global getroffen.
"""


def fitness(encoded, schedule):
    solution = decode(encoded, schedule)

    # Bei Zeitfensterüberschneidungen ist die Bewertung = unendlich, da dies hc2 verletzt.
    if is_overlapping(solution):
        return np.inf

    # Bewertungsverfahren
    if methodePlan == "sum of tardiness":
        scores = [slot["timestamp"] - slot["desired_timestamp"] for i, slot in solution.iterrows()]
        scores = np.abs(scores)
        score = sum(scores)
    elif methodePlan == "mean of tardiness":
        scores = [slot["timestamp"] - slot["desired_timestamp"] for i, slot in solution.iterrows()]
        scores = np.abs(scores)
        score = sum(scores) / len(solution)
    elif methodePlan == "tardy orders":  # 1: tardy, 0: not tardy
        scores = [slot["timestamp"] - slot["desired_timestamp"] for i, slot in solution.iterrows()]
        scores = np.abs(scores)
        scores = [1 for i in scores if i != 0]  # todo
        score = len(scores)
    elif methodePlan == "sum of quad tardiness":
        scores = [slot["timestamp"] - slot["desired_timestamp"] for i, slot in solution.iterrows()]
        scores = np.abs(scores)
        scores = [i * i for i in scores]
        score = sum(scores)

    return score


"""
Neue Methode:
Berechnet die Bewertung während der reaktiven Optimierung.
"""


def reaction_fitness(encoded, newschedule, oldschedule):
    tardiness = None
    solution = decode(encoded, newschedule)

    # Bei Zeitfensterüberschneidungen ist die Bewertung = unendlich, da dies hc2 verletzt.
    if is_overlapping(solution):
        return np.inf

    tardiness = fitness(encoded, newschedule)  # nutzt das prädiktive Bewertungsverfahren zur Bestimmung des Verzugs
    ef_list = (oldschedule['timestamp'] - solution['timestamp']).abs() / newschedule['duration']  # Berechnet die Differenz zweier Pläne
    erhaltungsfaktor = sum(ef_list)
    bewertung = tardiness + weight * erhaltungsfaktor  # Formel der Bewertungsfunktion
    return bewertung

"""
Neue Methode:
Aktualisiert die beste Bewertung, wenn die übergeben Bewertung besser ist.
"""


def update_currentbestscore(score):
    global currentBestScore
    if currentBestScore is None or score < currentBestScore:
        currentBestScore = score

    return currentBestScore


"""----------------------------------------Config-----------------------------------------"""
"""
In diesem Abschnitt werden globale Einstellungen vorgenommen.
"""



# Wähle den Datensatz aus: "lager_gross", "lager_mittel", "lager_klein"
DATASET = "lager_klein"
# Setze Speicherort
PATH = r"C:\Users\cemye\OneDrive - uni-oldenburg.de\Bachelorarbeit\reactive ZMS"

# Wähle Bewertungsfunktion für Gesamtplan: "sum of tardiness", "mean of tardiness, tardy orders, sum of quad tardiness
methodePlan = "mean of tardiness"
# Wähle Grenze für Neuplanung: 1 ist gleichbleibende Qualität, Verschlechterung bedeutet scoreThreshold > 1
scoreThreshold = 2
# Sollen Diagramme in der IDE angezeigt werden? - Boolean
show_plots = False
# Sollen Diagramme bei der Optimierung generiert werden? - Boolean
generate_opt_plots = False
# Sollen Diagramme gespeichert werden? - Boolean
save_plots = True

# Gewicht für den Erhaltungsfakore der Bewertungsfunktion
weight = 0



"""-----------------------------------------Setup-----------------------------------------"""
"""
In diesem Abschnitt wird werden Vorbereitungen getroffen
Zu Beginn werden die Zeitfensteranfragen und die Ereignisse aus dem Repository als JSON Datein geladen und eingelesen.
"""

with open(f"{PATH}/{DATASET}.json") as file:
    problem = json.load(file)
with open(f"{PATH}/ereignisse.json") as file2:
    events = json.load(file2)

pd.set_option('mode.chained_assignment', None)
orders = pd.DataFrame(problem["orders"], index=[x for x in range(len(problem["orders"]))])
added_ramps = pd.DataFrame(events["add_ramp"])
deleted_ramps = pd.DataFrame(events["delete_ramp"])
added_slots = pd.DataFrame(events["add_slot"])
deleted_slots = pd.DataFrame(events["delete_slot"])
delayed_slots = pd.DataFrame(events["delay_slot"])
maintainance = pd.DataFrame(events["maintainance"])
events = pd.DataFrame()
currentBestScore = None
plot_number = 0
decoded = None
list_of_deleted_ramps = []  # zur Überprüfung bei der Optimierung ob Rampen noch existieren

"""Die Funktionen für die Vorbereitung der Daten werden aufgerufen und abgespeichert."""

preprocessed_added_ramps = preprocess_added_ramps(added_ramps)
preprocessed_added_slots = preprocess_added_slots(added_slots)
preprocessed_deleted_ramps = preprocess_deleted_ramps(deleted_ramps)
preprocessed_deleted_slots = preprocess_deleted_slots(deleted_slots)
preprocessed_delayed_slots = preprocess_delayed_slots(delayed_slots)
preprocessed_maintainance = preprocess_maintainance(maintainance)

"""Alle Zeitstempel der Ereignisse sowie die Ereignisart **type** werden einer Event Tabelle hinzugefügt und angezeigt.
Diese Tabelle wird im weiteren Verlauf zum Ausfüren der Ereignis in der richtigen Reihenfolge benötigt, momentan muss jedoch zuerst der Plan erstellt werden."""

events["timestamp"] = [
    *preprocessed_added_slots["timestamp"],
    *preprocessed_deleted_slots["timestamp"],
    *preprocessed_added_ramps["timestamp"],
    *preprocessed_deleted_ramps["timestamp"],
    *preprocessed_delayed_slots["timestamp"],
    *preprocessed_maintainance["timestamp"]
]

events["type"] = [
    *preprocessed_added_slots["type"],
    *preprocessed_deleted_slots["type"],
    *preprocessed_added_ramps["type"],
    *preprocessed_deleted_ramps["type"],
    *preprocessed_delayed_slots["type"],
    *preprocessed_maintainance["type"]
]

events = events.sort_values(by=['timestamp'])

orders_processed = preprocess(orders, problem["number_ramps"])

"""----------------------------------------Ablauf----------------------------------------"""
"""
In diesem Abschnitt werden die Methoden nach dem Ablauf ausgeführt.
Neue Methode:
"""

"""Beginn: prädiktive Ablaufplanung"""
print("initialer Ablaufplan erstellen...")
for schedule in delay_fit(orders_processed, problem["number_ramps"]):
    fig, gnt = plot(schedule, problem["number_ramps"])
    showPlot()
print("Erstellung beendet")

newest_schedule = schedule.copy(deep=True)
newest_schedule = newest_schedule.astype('int32')
newest_schedule["timestamp"] = newest_schedule["timestamp"].astype("datetime64[s]")
newest_schedule["desired_timestamp"] = newest_schedule["desired_timestamp"].astype("datetime64[s]")
newest_schedule["endtimes"] = newest_schedule["endtimes"].astype("datetime64[s]")
newest_schedule["duration"] = newest_schedule["duration"] // 60

optimize(schedule, False)

newest_schedule = decoded.copy(deep=True)
newest_schedule = newest_schedule.astype('int32')
newest_schedule["timestamp"] = newest_schedule["timestamp"].astype("datetime64[s]")
newest_schedule["desired_timestamp"] = newest_schedule["desired_timestamp"].astype("datetime64[s]")
newest_schedule["endtimes"] = newest_schedule["endtimes"].astype("datetime64[s]")
newest_schedule["duration"] = newest_schedule["duration"] // 60
"""Ende: reaktive Ablaufplanung"""


"""Eine Liste mit den bisher gelöschten Rampen wird hier erstellt."""

deleted_ramps = []
decoded = decoded.astype(float)

"""Eine Ausgabe der Ereignisse in der Reihenfolge, in der sie ausgeführt werden. Die Ereignisse wurden zuvor basierend auf der mitgegebenen Ausführungszeit sortiert."""

events_output = events.copy(deep=True)
events_output["timestamp"] = events_output["timestamp"].astype("datetime64[s]")

"""
Die Event Tabelle wird durchlaufen, an *event_queue* übergeben und der erstellte Plan dargestellt, sowie der angepasste Plan in Tabellenform.
"""

print("Ereignisse go...")
new_schedule2 = decoded

#  Liste zur Bewertung wird erstellt.
eval = pd.DataFrame(events["type"])
eval.insert(1, "fitness before", None)
eval.insert(2, "fitness after", None)
eval.insert(3, "optimized?", None)

# Events werden chronologisch durchlaufen
for event_type in range(len(events["type"])):
    eval["fitness before"][event_type] = fitness(encode(new_schedule2), new_schedule2)
    new_schedule2 = event_queue(new_schedule2, event_type)
    fit = fitness(encode(new_schedule2), new_schedule2)
    eval["fitness after"][event_type] = fit
    update_currentbestscore(fit)
    fig, gnt = plot(new_schedule2, problem["number_ramps"])
    gnt.set_title(events["type"].iloc[event_type] + ", fitness = " + str(np.round(fit, 2)))
    showPlot()

    # Überprüft die Bewertung und optimiert, wenn scoreThreshold überschritten wird
    if fit != 0 and need_optimization(fit):
        fit = fitness(encode(new_schedule2), new_schedule2)
        optimize(new_schedule2, True)  # reaktive Optimierung
        new_schedule2 = decoded
        fig, gnt = plot(new_schedule2, problem["number_ramps"])
        gnt.set_title(events["type"].iloc[event_type] + ", fitness = " + str(np.round(fit, 2)))
        showPlot()
        eval["optimized?"][event_type] = True
    else:
        eval["optimized?"][event_type] = False

newest_schedule = new_schedule2.copy(deep=True)
newest_schedule = newest_schedule.astype('int32')
newest_schedule["timestamp"] = newest_schedule["timestamp"].astype("datetime64[s]")
newest_schedule["desired_timestamp"] = newest_schedule["desired_timestamp"].astype("datetime64[s]")
newest_schedule["endtimes"] = newest_schedule["endtimes"].astype("datetime64[s]")
newest_schedule["duration"] = newest_schedule["duration"] // 60
newest_schedule["score"] = newest_schedule["score"] // 60

print(eval.sort_index())
print("Gewicht des Erhaltungsfaktor = " + str(weight))
print("Score Threshold =" + str(scoreThreshold))

print("Ereignisse done")

"""-----------------------------------------Ende-----------------------------------------"""
