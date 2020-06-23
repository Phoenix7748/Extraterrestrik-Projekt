# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:12:50 2020

@author: Phoen
"""
# %% Importierte Module
import numpy as np
# numpy für arrays und vektorrechnung
import matplotlib.pyplot as plt
# matplotlib fürs plotten
from mpl_toolkits.mplot3d import Axes3D
# axes3d für das 3D-Plotten
from scipy import constants as const
# scipy für die (physikalischen) Konstanten

# %% Globale Konstanten, wird dank scipy nicht wirklich gebraucht, ist aber gut
"""Bei scipy haben wir als Elementarladung: e, elementary_charge
   Als Lichtgeschwindigkeit: c, speed_of_light
   Als die Permittivität des Vakuums: epsilon_0
   Als die Permeabilität des Vakuums: mu_0
   Als Elektronenmasse: m_e, electron_mass
   Als Protonenmasse: m_p, proton_mass
   Als Neutronenmasse: m_n, neutron_mass
   Als Pi: pi
   Und noch vieles mehr."""
pi = const.pi
# print("pi:", pi)
eps_0 = const.epsilon_0  # 8.854188E-12  # Elektrische Feldkonstante in As/Vm
# print("Permittivität des Vakuums:", eps_0)
k = 1/(4*pi*eps_0)     # Vorfaktor des S.I. in Vm/As
# print("S.I.-Konstante:", k)  # ist korrekt
c = const.c  # 299792458  # Lichtgeschwindigkeit in m/s
# print("Lichtgeschwindigkeit:", c)
mu_0 = const.mu_0  # 1/(eps_0*(c**2))  # Magnetische Feldkonstante in N/(A**2)
# print("Permeabilität des Vakuums:", mu_0)  # ist auch korrekt

# Zum Debugging
# pi = 1
# eps_0 = 1  # 8.854188E-12  # Elektrische Feldkonstante in As/Vm
# k = 1     # Vorfaktor des S.I. in Vm/As
# c = 1  # 299792458  # Lichtgeschwindigkeit in m/s
# mu_0 = 1  # 1/(eps_0*(c**2))  # Magnetische Feldkonstante in N/(A**2)

# %% Beschreibung des Programmes, Ideen und Benutzungshinweise
# Projekt 6, Helixplot (Gyrationsbahnen)

"""Mit numerischen Verfahren soll die Trajektorie eines (beliebigen) Teilchens
   in einem (beliebigen) Magnetfeld bestimmt und 3d-geplottet werden."""

"""Singlaritäten sind zur Zeit noch ein Problem.(vlt behoben)
   Zurzeit läuft das Programm ohne Laufzeitfehler, aber gibt Unsinn aus.
   Für ein homogenes B-Feld wird zurzeit kein Kreis, sondern ein Spirale
   ausgegeben, und eine Geschwindigkeitskomponente parallel zu B sorgt auch
   noch für merkwürdige Ergebnise, statt Helix gibt das Programm eine Spirale
   auf einer Stelze aus.
   """

# %% Klassen

# """Klasse der Magnetfelder -  diese sind (zeitabhängige) Vektorfelder"""


# class magnetic_field():
#     pass


# """Klasse der elektrischen Felder-diese sind (zeitabhängige) Vektorfelder"""


# class electric_field():
#     pass


"""Klasse der Teilchen. Diese haben Massen, Ladungen, Spins, Position
   (als kartesischen Ortsvektor) und Geschwindigkeit (auch als kart. Vektor)"""


class Particle():

    def __init__(self, mass, charge, spin, position, velocity, time):
        self.mass = mass          # float in Kilogramm
        self.charge = charge      # float in Elementarladungen
        self.spin = spin          # float
        self.position = np.array(position, dtype="float64")  # numpy.array in Metern
        self.velocity = np.array(velocity, dtype="float64")  # numpy.array in Metern
        self.time = time

    def move(self, delta_t):
        # np.add(self.position, self.velocity*delta_t,
        #        out=self.position, casting="unsafe")
        # Der obige Ausdruck entspricht dem Unteren, ist aber nötig, da
        # sich arrays aus int32 und arrays aus float64 scheinbar sonst nicht
        # addieren lassen
        self.position += self.velocity*delta_t  # geht dank typecast doch
        self.time += delta_t
        # self.time = np.add(self.time, delta_t, casting="unsafe")
        # np.add(self.time, delta_t, out=self.time, casting="unsafe")
        # hier musste aus selben Grund wie oben add benutzt werden.
        # self.time += delta_t geht leider nicht
        # np.add(self.time, delta_t, out=self.time, casting = "unsafe")
        # geht auch nicht da out ein array seien muss

    def accelerate(self, delta_v):
        # np.add(self.velocity, delta_v, out=self.velocity, casting="unsafe")
        # wie bei move, muss auch hier add benutzt werden
        self.velocity += delta_v  # dank typecast auf float64 nun nutzbar


"""Ein paar Tochterklassen zu particles, die jeweils eine Teilchenart
   repräsentieren"""


class Electron(Particle):

    def __init__(self, position, velocity, time):
        self.mass = const.m_e   # 9.109384E-31   # float in Kilogramm
        self.charge = -const.e  # -1.602177E-19  # float in Coulomb
        self.spin = 0.5                          # float
        self.position = np.array(position, dtype="float64")  # numpy.array in Metern
        self.velocity = np.array(velocity, dtype="float64")  # numpy.array in Metern
        self.time = time


class Proton(Particle):

    def __init__(self, position, velocity, time):
        self.mass = const.m_p  # 1.672622E-27    # float in Kilogramm
        self.charge = const.e  # 1.602177E-19    # float in Coulomb
        self.spin = 0.5                          # float
        self.position = np.array(position, dtype="float64")  # numpy.array in Metern
        self.velocity = np.array(velocity, dtype="float64")  # numpy.array in Metern
        self.time = time


class Neutron(Particle):

    def __init__(self, position, velocity, time):
        self.mass = const.m_n  #                  # float in Kilogramm
        self.charge = 0                           # float in Coulomb
        self.spin = 1  # ?, gerade nicht wichtig  # float
        self.position = np.array(position, dtype="float64")  # numpy.array in Metern
        self.velocity = np.array(velocity, dtype="float64")  # numpy.array in Metern
        self.time = time


"""Klasse der Zellen. Bei denen handelt es sich um Objekte die als Attribute
   einen Mittelpunkt, 3 Intervalle und 2 Felder haben"""


class CartesianCell():

    def __init__(self, center, x_intvl, y_intvl, z_intvl, mag_fld, elec_fld):
        self.center = center                             # numpy.array
        self.x_start = x_intvl[0]                        # float
        self.x_end = x_intvl[1]                          # float
        self.y_start = y_intvl[0]                        # float
        self.y_end = y_intvl[1]                          # float
        self.z_start = z_intvl[0]                        # float
        self.z_end = z_intvl[1]                          # float
        self.mag_fld =  lambda t : mag_fld(center, t)    # function(time)
        self.elec_fld =  lambda t : elec_fld(center, t)  # function(time)

    def copy(self):
        cntr = self.center
        x_intvl = (self.x_start, self.x_end)
        y_intvl = (self.y_start, self.y_end)
        z_intvl = (self.z_start, self.z_end)
        mag = self.mag_fld
        elec = self.elec_fld
        return CartesianCell(cntr, x_intvl, y_intvl, z_intvl, mag, elec)

"""Klasse der kartesischen Räume. Diese bestehen aus einer Matrix Z aus Zellen.
   Ich gehe davon aus, dass die Tupel x,y,z_tripel jeweils den Anfangs- und
   Endwert sowie auch die Schrittweite der Werte in den jeweiligen Achsen
   angibt(Bsp.: (3, 24, 1.5))."""


class CartesianSpace():

    """Für jeden diskreten x-Werte lege ich eine Zeile in dieser Matrix an
       für jeden y-Wert eine Spalte und für jeden z-Werte eine "Reihe".
       die Elemente dieser Matrix sind Zellen aus x-,y- und z-Intervallen
       die somit wieder den gesamten R^3 füllen.
       Die zellen sind so angeorndet, dass wenn man sich nur in positive
       x-Richtung bewegt nur der erste Index steigt während der y- und der
       z-Index fixiert bleiben. Analoges für Bewegungen in andere Richtungen"""

    def __init__(self, x_tripel, y_tripel, z_tripel, mag_fld, elec_fld):
        x_stride = x_tripel[2]
        x_arr = np.arange(x_tripel[0], x_tripel[1], x_tripel[2])
        y_stride = y_tripel[2]
        y_arr = np.arange(y_tripel[0], y_tripel[1], y_tripel[2])
        z_stride = x_tripel[2]
        z_arr = np.arange(z_tripel[0], z_tripel[1], z_tripel[2])
        cell_lol = []
        x_index = 0
        for x in x_arr:
            cell_lol.append([])
            y_index = 0
            print("Ich lege die {0}.te Zeile an!".format(x_index+1))
            for y in y_arr:
                cell_lol[x_index].append([])
                for z in z_arr:
                    center = np.array([x, y, z])            # numpyarray
                    x_intvl = (x-x_stride/2, x+x_stride/2)  # Tupel
                    y_intvl = (y-y_stride/2, y+y_stride/2)  # Tupel
                    z_intvl = (z-z_stride/2, z+z_stride/2)  # Tupel
                    new_cell = CartesianCell(center, x_intvl, y_intvl, z_intvl,
                                             mag_fld, elec_fld)
                    cell_lol[x_index][y_index].append(new_cell)
                y_index += 1
            x_index += 1
        # print(cell_lol)
        self.matrix = np.array(cell_lol)
        self.x_arr = x_arr
        self.x_stride = x_stride
        self.y_arr = y_arr
        self.y_stride = y_stride
        self.z_arr = z_arr
        self.z_stride = z_stride

# Spacetime verbindet einen CartesianSpace mit einer Zeit, somit ließe sich
# der selbe Raum für mehrere Teilchen nutzen, deren plots jeweils andere
# Spacetimes nutzen


class SpaceTime():
    pass

# %% Funktionen


"""find_first_cell soll für eine gegebene Position die Zelle in der
   Zellenmatrix: Z finden. Diese Funktion ließe sich nutzen um für jede
   Position die Zelle zu bestimmen, doch für alle Positionen nach der
   ersten lassen sich die Zelle aus der aktuellen Zelle und der
   Bewegungsrichtung deduzieren.
   Die Funktion gibt jedoch nicht die gesuchte Zelle sondern den Index heraus,
   den diese Zelle in der Matrix hätte. Wenn einer der Indizes negativ seien
   sollte oder über die Grenzen der Matrix hinausgeht, dann liegt der Punkt
   außerhalb des Raumes den die Zellen der Matrix abdecken."""


def find_first_cell(position, space):
    matrix_zero = space.matrix[0][0][0].center-np.array([space.x_stride/2,
                                                         space.y_stride/2,
                                                         space.z_stride/2])
    cell_x_ind = int((position[0]-matrix_zero[0])//space.x_stride)  # int
    cell_y_ind = int((position[1]-matrix_zero[1])//space.y_stride)  # int
    cell_z_ind = int((position[2]-matrix_zero[2])//space.z_stride)  # int
    cell_index = (cell_x_ind, cell_y_ind, cell_z_ind)               # tupel
    print("Die erste Zelle hat die Indizes: ", cell_index)
    return (cell_index)


"""Dies Funktion find_next_cell nimmt ein Teilchen, einen Raum aus Zellen,
   sowie auch das Index-Tupel der aktuellen Position und gibt das Index-Tupel
   der nächsten Zelle aus, sowie auch die Zeit dt die benötigt wird um in die
   nächste Zelle zu gelangen."""


def find_next_cell(particle, space, indx):
    # print("find_next_cell")
    mtx = space.matrix
    v = particle.velocity
    # Bestimme das x_limit und die Zeit x_time zu diesem
    if v[0] == 0:
        x_exit = 0
        x_time = np.Infinity
        # print("x-Geschwindigkeit=0")
    else:
        if v[0] >= 0:
            x_limit = mtx[indx[0]][indx[1]][indx[2]].x_end
            x_exit = 1
        else:
            x_limit = mtx[indx[0]][indx[1]][indx[2]].x_start
            x_exit = -1
        x_time = abs((x_limit-particle.position[0])/v[0])
    # Bestimme das y_limit und die Zeit y_time zu diesem
    if v[1] == 0:
        y_exit = 0
        y_time = np.Infinity
        # print("y-Geschwindigkeit=0")
    else:
        if v[1] >= 0:
            y_limit = mtx[indx[0]][indx[1]][indx[2]].y_end
            y_exit = 1
        else:
            y_limit = mtx[indx[0]][indx[1]][indx[2]].y_start
            y_exit = -1
        y_time = abs((y_limit-particle.position[1])/v[1])
    # Bestimme das y_limit und die Zeit y_time zu diesem
    if v[2] == 0:
        z_exit = 0
        z_time = np.Infinity
        # print("z-Geschwindigkeit=0")
    else:
        if v[2] >= 0:
            z_limit = mtx[indx[0]][indx[1]][indx[2]].z_end
            z_exit = 1
        else:
            z_limit = mtx[indx[0]][indx[1]][indx[2]].z_start
            z_exit = -1
        z_time = abs((z_limit-particle.position[2])/v[2])
    # Erhöhe einen Index um 1
    if x_time <= y_time and x_time <= z_time:
        # print("Ich erhöhe x-index")
        dt = x_time
        indx = (indx[0]+x_exit, indx[1], indx[2])
    elif y_time <= z_time:
        # print("Ich erhöhe y-index")
        dt = y_time
        indx = (indx[0], indx[1]+y_exit, indx[2])
    else:
        # print("Ich erhöhe z-index")
        dt = z_time
        indx = (indx[0], indx[1], indx[2]+z_exit)
    return(indx, dt)


"""check_index nimmt ein Index-Tupel und einen Raum und überprüft, ob
   die durch das Tupel indizierte Zelle im Raum liegt und gibt dementsprechend
   True oder False aus."""


def check_index(indx, space):
    x_len = len(space.matrix)
    y_len = len(space.matrix[0])
    z_len = len(space.matrix[0][0])
    if indx[0] < 0 or indx[1] < 0 or indx[2] < 0:
        return False
    elif indx[0] >= x_len or indx[1] >= y_len or indx[2] >= z_len:
        return False
    else:
        return True


"""trajectory soll die Trajektorie eines Teilchens in den Feldern bestimmen.
   dafür speichert die Funktion in einer Liste alle Zellen die durchschritten
   werden, alle Zellendurchstoßpunkte (einschließlich des Startpunktes)
   und die jeweiligen Geschwindigkeiten, vlt auch die Zeitpunkte.
   mtx ist nur ein Kürzel für die Zellenmatrix des Raumes: space.
   indx ist ein 3-elementiges Tupel welches dazu dient die aktuelle Zelle
   zu indizieren. in_space ist eine Flag, die angibt ob sich die aktuelle
   Position noch in space befindet. x_exit gibt an, ob die Zelle in
   positiver oder negativer x-Richtung verlassen wurde, analoges gilt für
   y_exit und z_exit.
   Die verschiedenen __time variablen speichern die Zeit, die das Teilchen
   benötigen würde um im Falle einer Bewegung parallel zur _-Achse die Zelle
   zu verlassen. Das Minimum dieser Zeiten gibt uns an, durch welche Wand die
   Zelle wann verlassen wird. Die somit bestimmte Zeit kann dann auf die
   aktuelle Zeit des Teilchens gerechnet werden und die neue Position des
   Teilchens wird durch diese Zeit und die aktuelle Geschwindigkeit des
   Teilchens bestimmt. Desweiteren wird die neue Geschwindigkeit des Teilchens
   durch die Felder, die Zeit die Ladung und die Masse bestimmt."""


def trajectory(particle, space):
    q = particle.charge  # float
    m = particle.mass    # float
    mtx = space.matrix  # numpy.array
    indx = find_first_cell(particle.position, space)  # Tupel(bzw. ein Tripel)
    v = particle.velocity  # numpy.array
    cell = mtx[indx[0]][indx[1]][indx[2]]  # CartesianCell
    cell_list = []
    position_list = []
    velocity_list = []
    time_list = []
    in_space = check_index(indx, space)  # boolean
    count = 0
    while in_space and count <= 10000:
        print("count ist: ", count)  # inkrementiert immer um 1, auch gut
        # print("Index ist: ", indx)  # geht immer brav nur einen weiter
        cell_list.append(cell.copy())
        position_list.append(particle.position.copy())
        velocity_list.append(v.copy())
        time_list.append(particle.time)
        lorentz_acc = (q/m)*(cell.elec_fld(particle.time)+np.cross(v, cell.mag_fld(particle.time)))
        print("lorentz_acc: ", lorentz_acc)
        indx, dt = find_next_cell(particle, space, indx)
        # print("dt: ", dt)
        particle.move(dt)
        print("particle.position: ", particle.position)
        particle.accelerate(lorentz_acc*dt)
        print("particle.velocity: ", particle.velocity)
        count += 1
        in_space = check_index(indx, space)  # boolean
    return(cell_list, position_list, velocity_list, time_list)


"""plot_trajectory soll eine Trajektorie plotten. Sollen die Feldlinie,
   insbesondere die "Führungslinie"(?) mit geplottet werden?
   ließe sich das auch als gif animieren? Ließen sich auch mehrere Teilchen
   in einem Raum plotten?"""


def plot_trajectory():
    pass


def plot_mag_field():
    pass


def plot_elec_field():
    pass


# %% Daten generieren und einlesen

"""Je nachdem wie die Felder und Teilchen implementiert werden, lohnt es
   sich womöglich diese (wenn es sich um größere Datensätze handelt)
   zu generieren, in Dateien zu speichern und dann wieder einzulesen"""


# %% Main

"""Wenn wir nun unsere Felder und Teilchen haben werden hier die Trajektorien
   bestimmt und ausgegeben."""

# Elektrische Feldstärke um eine Punktladung (+1C) im Nullpunkt
# elec_field = lambda r, t: -(k/np.sqrt(r[0]**2+r[1]**2+r[2]**2)**3)*r+0*t


def elec_field1(r, t):
    q = 1  # Ladung 1 C
    length = (k*(q+0*t))/(np.sqrt(r[0]**2+r[1]**2+r[2]**2)**2)
    direction = (r/np.sqrt(r[0]**2+r[1]**2+r[2]**2))
    return length*direction


# kein E-Feld
def elec_field2(r, t):
    return 0*r*t


# Magnetisches Feld um einen Leiter (1A) auf der z-Achse
# mag_field = lambda r, t: (mu_0/(2*np.pi*np.sqrt(r[0]**2+r[1]**2)**2))
# *np.cross(np.array([0, 0, 1]), r)+0*t


def mag_field1(r, t):
    i = 1  # Stromstärke 1 A
    length = (mu_0*(i+0*t)/(2*np.pi*np.sqrt(r[0]**2+r[1]**2)))
    direction = np.array([-r[1], r[0], 0])/np.sqrt(r[0]**2+r[1]**2)
    return length*direction


# konstantes Magnetfeld parallel zur z-Achse
def mag_field2(r, t):
    return np.array([0, 0, 0.01])


# konstantes Magnetfeld parallel zur y-Achse
def mag_field3(r, t):
    return np.array([0, 0.01, 0])


x_tripel = (-5, 5, 0.1)
y_tripel = (-5, 5, 0.1)
z_tripel = (-5, 5, 0.1)

print("Ich beginne den Raum zu erstellen!")
space1 = CartesianSpace(x_tripel, y_tripel, z_tripel, mag_field2, elec_field2)
print("Raum ist fertiggestellt!")

# print(np.shape(space1.matrix))
# print(np.shape(space1.matrix[0]))
# print(np.shape(space1.matrix[1]))
# print(space1.matrix[0, 0, 0].center)
# print(space1.matrix[0, 0, 0].elec_fld(3))
# print(space1.matrix[0, 0, 0].elec_fld(6))

# %% Hier erstelle ich die Partikel und setzte diese ein

position1 = np.array([0, 1, 0])
velocity1 = np.array([0, 0, 1])
time1 = 0
position2 = np.array([0, 0, 0])
velocity2 = np.array([0, -10, 1000000])
time2 = 0
position3 = np.array([0, 0, 10])
velocity3 = np.array([0, 10000, 1000])
time3 = 0


electron1 = Electron(position1, velocity1, time1)
electron2 = Electron(position2, velocity2, time2)
electron3 = Electron(position3, velocity3, time3)
proton1 = Proton(position1, velocity1, time1)
proton2 = Proton(position2, velocity2, time2)
neutron1 = Neutron(position1, velocity1, time1)
neutron2 = Neutron(position2, velocity2, time2)

print("Ich fange an die Trajektorien zu bestimmen!")
tupel_e = trajectory(electron2, space1)
# tupel_p = trajectory(proton2, space1)
# tupel_n = trajectory(neutron2, space1)

print("Ich habe die Trajektorien bestimmt!")
positions_e = np.array(tupel_e[1]).T
# positions_p = np.array(tupel_p[1]).T

# zum Plotten ist das Transponieren notwendig, um die x-,y-,z-werte zu trennen
# positions_n = np.array(tupel_n[1]).T
# print(positions_e)

# print("\nZellen:\n", tupel_n[0])
# print("\nPositionen:\n", tupel_e[1])
# print("\nGeschwindigkeiten:\n", tupel_e[2])
print("\nZeiten\n", tupel_e[3][-1])
# %% Plotten
fig = plt.figure(figsize=(30, 30))
ax = fig.gca(projection='3d')

print("Ich fange an zu plotten!")
ax.plot(positions_e[0], positions_e[1], positions_e[2], lw=0.5, color="b")
# ax.plot(positions_p[0], positions_p[1], positions_p[2], lw=0.5, color="r")
# ax.plot(positions_n[0], positions_n[1], positions_n[2], lw=0.5, color="k")

print("Ich bin fertig mit dem Plotten!")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Helix-Plot")
ax.view_init(55, 30)

plt.show()
fig.savefig("Helixplot")

# %% Testing

"""Hier wird sich an Neuem und Vergessenem versucht."""

# checke die Teilchenklassen
# elec_position = np.array([5, 2, 6.3])
# elec_velocity = np.array([10, 0, 0])
# electron = Electron(elec_position, elec_velocity, 0)
# electron.move(1)
# electron.accelerate(np.array([0, 5, 0]))
# print(electron.position)
# print(electron.velocity)

# checke wie sich Funktionen höherer Ordnung nutzen lassen um aus einem
# Orts- und zeitabhänigen Feld viele zeitabhängige Felder an ver. Orten zu gen.
# time = 1
# elec_field = lambda r, t : t*r[0]
# elec_field_pos1 = lambda t : elec_field(electron.position, t)
# print(elec_field(electron.position, time))
# print(elec_field_pos1(time))
# time = 2
# print(elec_field_pos1(time))


# # Beispiel x,y,z arrays
# x_arr1 = np.linspace(0, 99, 100)
# temp = []
# for i in range(100):
#     temp.append(np.random.randint(0, 100))
# temp = np.array(sorted(temp))
# temp2 = []
# for i in temp:
#     if i not in temp2:
#         temp2.append(i)
# x_arr2 = np.array(temp2)
# print("x_arr1:\n", x_arr1)
# print("\n")
# print("x_arr2:\n", x_arr2)
# print("\n")

# y_arr1 = np.linspace(50, 149, 100)
# temp = []
# for i in range(100):
#     temp.append(np.random.randint(50, 149))
# temp = np.array(sorted(temp))
# temp2 = []
# for i in temp:
#     if i not in temp2:
#         temp2.append(i)
# y_arr2 = np.array(temp2)
# print("y_arr1:\n", y_arr1)
# print("\n")
# print("y_arr2:\n", y_arr2)
# print("\n")

# z_arr1 = np.linspace(-50, 49, 100)
# temp = []
# for i in range(100):
#     temp.append(np.random.randint(-50, 49))
# temp = np.array(sorted(temp))
# temp2 = []
# for i in temp:
#     if i not in temp2:
#         temp2.append(i)
# z_arr2 = np.array(temp2)
# print("z_arr1:\n", z_arr1)
# print("\n")
# print("z_arr2:\n", z_arr2)


# Beispiel-Felder:


# print(find_first_cell(position1, space1))

# Kreuzproduktest
# arr1 = np.array([0, 1, 2])
# arr2 = np.array([3, 3, 3])
# print(np.cross(arr1, arr2))

# np.array-dynamische Erweiterung-Test... geht nicht^^
# arr3 =np.array([1, 2, 3])
# arr3[3] = 4
# print(arr3)

# a = np.array([3234324.1231241234, 3234324.1231241234, 3234324.1231241234])
# print(type(a))
# print(type(a[0]))
# b = np.array([2333333, 2333333, 2333333])
# print(type(b))
# print(type(b[0]))
# a += b
# print(a)
# np.add(a, b, out=a, casting="unsafe")
# print(a)
# a = np.add(a, b, casting="unsafe")
# print(a)

# x = 3234324.1231241234
# y = 2333333
# # np.add(x, y, out=x, casting="unsafe")
# x = np.add(x, y, casting="unsafe")
# print(x)
# print(np.pi)

# position1 = np.array([0, 1, 0])
# velocity1 = np.array([0, 1, 0])
# position2 = np.array([0, -10, 0])
# velocity2 = np.array([0, -1, 0])


# def lorentz_acc(v, t, q, m, r):
#     acc = (q/m)*(elec_field2(r, t)+np.cross(v, mag_field2(r, t)))
#     print("Kreuzprodukt vxB: ", np.cross(v, mag_field2(r, t)))
#     return acc


# m = 9.109384E-31     # float in Kilogramm
# q = -1.602177E-19  # float in Coulomb

# print(position1[0])
# print(position1[0]**2)
# print(position1[1])
# print(position1[1]**2)
# print(position1[2])
# print(position1[2]**2)
# print(elec_field2(position1, 0))
# print(mag_field2(position1, 0))
# print(elec_field2(position2, 0))
# print(mag_field2(position2, 0))
# print("lorentz-beschleunigung1: ", lorentz_acc(velocity1, 0, q, m, position1))
# print("lorentz-beschleunigung1: ", lorentz_acc(velocity2, 0, q, m, position2))


# # checke reshape

# a = np.array([1, 2, 3, 4, 5, 6])
# print(np.shape(a))
# b = np.reshape(a, (3, 2))
# print(b)
# c = np.reshape(b, (2, 3))
# print(c)

# a_list = [0, 1, 2, 3]
# arr1 = np.array(a_list, dtype="float32")
# print(arr1)
# arr2 = np.array(a_list, dtype="int32")
# print(arr2)