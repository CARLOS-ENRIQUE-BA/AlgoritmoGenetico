import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify
import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import imageio.v2 as imageio  

# Función de aptitud
def fitness_function(x):
    return np.log(1 + x + 50 * np.abs(x)) + 5 * np.sin(x)

# Calcular el valor de la función
def calculate_function(func, x_value):
    x = symbols('x')
    expression = lambdify(x, func, 'numpy')
    return expression(x_value)

# Calcular el valor de x a partir de un número binario
def binary_to_x(binary_value, lower_limit, delta):
    return lower_limit + binary_value * delta

# Crear nuevos individuos
def create_new_individuals(binary1, binary2, lower_limit, delta):
    decimal1 = int(binary1, 2)
    decimal2 = int(binary2, 2)
    x1 = binary_to_x(decimal1, lower_limit, delta)
    x2 = binary_to_x(decimal2, lower_limit, delta)
    y1 = fitness_function(x1)
    y2 = fitness_function(x2)

    new_individual1 = {'id': None, 'binary': binary1, 'decimal': decimal1, 'x': x1, 'y': y1}
    new_individual2 = {'id': None, 'binary': binary2, 'decimal': decimal2, 'x': x2, 'y': y2}

    return new_individual1, new_individual2

# Mutación de individuos por intercambio de bits
def mutate(individual_binary, mutation_prob_gen, mutation_prob_ind):
    binary_list = list(individual_binary)
    if random.random() < mutation_prob_ind:
        for i in range(len(binary_list)):
            if random.random() < mutation_prob_gen:
                pos2 = random.randint(0, len(binary_list) - 1)
                binary_list[i], binary_list[pos2] = binary_list[pos2], binary_list[i]
    return ''.join(binary_list)

# Cruce de individuos con múltiples puntos de cruza
def multiple_point_crossover(parent1, parent2, bit_length):
    num_crossover_points = random.randint(1, bit_length - 1)
    crossover_points = sorted(random.sample(range(1, bit_length), num_crossover_points))

    new_binary1 = list(parent1['binary'])
    new_binary2 = list(parent2['binary'])

    for i in range(len(crossover_points)):
        if i % 2 == 1:
            continue
        start = crossover_points[i]
        end = crossover_points[i + 1] if i + 1 < len(crossover_points) else bit_length
        new_binary1[start:end], new_binary2[start:end] = new_binary2[start:end], new_binary1[start:end]

    return ''.join(new_binary1), ''.join(new_binary2)

# Formar parejas según la estrategia A1
def pair_individuals(population):
    pairs = []
    for individual in population:
        num_mates = random.randint(1, len(population) - 1)
        mates = random.sample([ind for ind in population if ind != individual], num_mates)
        for mate in mates:
            pairs.append((individual, mate))
    return pairs

# Poda de la población, manteniendo los mejores (estrategia P1)
def prune_population(population, max_pop_size, problem_type):
    unique_population = {individual['decimal']: individual for individual in population}.values()
    sorted_population = sorted(unique_population, key=lambda x: x['y'], reverse=(problem_type == "maximizacion"))
    return list(sorted_population)[:max_pop_size]

# Generar la población inicial
def generate_initial_population(pop_size, value_range, bit_length, lower_limit, delta):
    population = []
    for i in range(pop_size):
        random_value = random.randint(0, value_range - 1)
        binary_value = bin(random_value)[2:].zfill(bit_length)
        x_value = binary_to_x(random_value, lower_limit, delta)
        y_value = fitness_function(x_value)
        individual = {'id': i + 1, 'binary': binary_value, 'decimal': random_value, 'x': x_value, 'y': y_value}
        population.append(individual)
    return population

# Calcular parámetros iniciales
def calculate_initial_params(lower_limit, upper_limit, resolution):
    range_value = upper_limit - lower_limit
    num_points = math.ceil(range_value / resolution) + 1
    bit_length = math.ceil(math.log2(num_points))
    delta = range_value / ((2 ** bit_length) - 1)
    value_range = 2 ** bit_length
    return delta, bit_length, value_range

# Graficar estadísticas de todas las generaciones
def plot_statistics(generations, best_y, worst_y, average_y):
    plt.figure()
    plt.plot(generations, best_y, label='Mejor Individuo')
    plt.plot(generations, worst_y, label='Peor Individuo')
    plt.plot(generations, average_y, label='Promedio')

    plt.title('Evolución del fitness')
    plt.xlabel('Generación')
    plt.ylabel('Valor de la Función Objetivo')
    plt.legend()

    folder_path = 'Media_Generaciones'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, 'IMG_Media_Generaciones.png'))
    plt.close()

# Graficar la población de una generación
def plot_generation(generation, population, lower_limit, upper_limit, problem_type):
    plt.clf()
    plt.xlim(lower_limit, upper_limit)
    plt.title(f'Generación {generation}')
    plt.xlabel('X')
    plt.ylabel('f(X)')

    x_values = [individual['x'] for individual in population]
    y_values = [individual['y'] for individual in population]

    plt.scatter(x_values, y_values, label="Individuos", s=90, c="#45aaf2", alpha=0.4)

    if problem_type == "maximizacion":
        best_individual = max(population, key=lambda individual: individual['y'])
        worst_individual = min(population, key=lambda individual: individual['y'])
    else:
        best_individual = min(population, key=lambda individual: individual['y'])
        worst_individual = max(population, key=lambda individual: individual['y'])

    x_func = np.linspace(lower_limit, upper_limit, 200)
    y_func = fitness_function(x_func)
    plt.plot(x_func, y_func)

    plt.scatter(best_individual['x'], best_individual['y'], c='green', label='Mejor Individuo', s=90)
    plt.scatter(worst_individual['x'], worst_individual['y'], c='red', label='Peor Individuo', s=90)

    plt.legend()

    folder_path = 'Generaciones'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, f'IMG_Generacion_{generation}.png'))
    plt.close()

# Crear el video de las generaciones en formato .mp4
def create_video_from_images(folder_path, output_file):
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))  # Ordenar imágenes por número de generación

    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    video = imageio.get_writer(output_file, fps=1, codec='libx264')

    for image in images:
        video.append_data(imageio.imread(os.path.join(folder_path, image)))

    video.close()

# Algoritmo genético principal
def genetic_algorithm(lower_limit, upper_limit, resolution, initial_pop_size, max_pop_size, num_generations, mutation_prob_ind, mutation_prob_gen, problem_type, table):
    delta, bit_length, value_range = calculate_initial_params(lower_limit, upper_limit, resolution)
    population = generate_initial_population(initial_pop_size, value_range, bit_length, lower_limit, delta)

    generations = []
    best_y = []
    worst_y = []
    average_y = []

    # Graficar e imprimir la población inicial (Generación 0)
    plot_generation(0, population, lower_limit, upper_limit, problem_type)
    best_individual = max(population, key=lambda individual: individual['y']) if problem_type == "maximizacion" else min(population, key=lambda individual: individual['y'])

    for generation in range(1, num_generations + 1):
        pairs = pair_individuals(population)
        new_population = []

        # Elitismo: conservar el mejor individuo de la generación anterior
        best_individual_previous_generation = best_individual
        #---------------------------------------------- implementar en esta parte el promedio ----------------------------------------------
        new_population.append(best_individual_previous_generation)

        for parent1, parent2 in pairs:
            child_binary1, child_binary2 = multiple_point_crossover(parent1, parent2, bit_length)

            child_binary1 = mutate(child_binary1, mutation_prob_gen, mutation_prob_ind)
            child_binary2 = mutate(child_binary2, mutation_prob_gen, mutation_prob_ind)

            child1, child2 = create_new_individuals(child_binary1, child_binary2, lower_limit, delta)
            new_population.append(child1)
            new_population.append(child2)

        population = prune_population(new_population, max_pop_size, problem_type)

        if problem_type == "maximizacion":
            best_individual = max(population, key=lambda individual: individual['y'])
            worst_individual = min(population, key=lambda individual: individual['y'])
        else:
            best_individual = min(population, key=lambda individual: individual['y'])
            worst_individual = max(population, key=lambda individual: individual['y'])

        generations.append(generation)
        best_y.append(best_individual['y'])
        worst_y.append(worst_individual['y'])
        average_y.append(sum(individual['y'] for individual in population) / len(population))

        plot_generation(generation, population, lower_limit, upper_limit, problem_type)

    plot_statistics(generations, best_y, worst_y, average_y)
    create_video_from_images('Generaciones', 'output_video.mp4')

    # Actualizar la tabla con los resultados del mejor individuo de la última generación
    table.delete(*table.get_children())
    table.insert("", "end", values=(best_individual['binary'], best_individual['decimal'], best_individual['x'], best_individual['y']))

# Interfaz gráfica
def run_gui():
    root = tk.Tk()
    root.title("Algoritmo Genético")
    root.geometry("800x600")

    parameters_frame = tk.Frame(root)
    parameters_frame.pack(pady=10)

    table_frame = tk.Frame(root)
    table_frame.pack(pady=10)

    labels = [
        "Límite Inferior:", "Límite Superior:", "Resolución:", "Tamaño Población Inicial:",
        "Tamaño Máximo Población:", "Número de Generaciones:", "Probabilidad de Mutación (Individuo):",
        "Probabilidad de Mutación (Gen):", "Tipo de Problema:"
    ]

    entries = {}
    for i, label in enumerate(labels):
        tk.Label(parameters_frame, text=label).grid(row=i, column=0, sticky="e")
        entry = tk.Entry(parameters_frame)
        entry.grid(row=i, column=1)
        entries[label] = entry

    problem_type = ttk.Combobox(parameters_frame, values=["maximizacion", "minimizacion"], state="readonly")
    problem_type.set("maximizacion")
    problem_type.grid(row=8, column=1)

    columns = ("Cadena de Bits", "Valor de la Cadena", "Valor de X", "Valor de f(X)")
    table = ttk.Treeview(table_frame, columns=columns, show="headings")
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=150)

    table.pack()

    def run_algorithm():
        try:
            lower_limit = float(entries["Límite Inferior:"].get())
            upper_limit = float(entries["Límite Superior:"].get())
            resolution = float(entries["Resolución:"].get())
            initial_pop_size = int(entries["Tamaño Población Inicial:"].get())
            max_pop_size = int(entries["Tamaño Máximo Población:"].get())
            num_generations = int(entries["Número de Generaciones:"].get())
            mutation_prob_ind = float(entries["Probabilidad de Mutación (Individuo):"].get())
            mutation_prob_gen = float(entries["Probabilidad de Mutación (Gen):"].get())
            problem_type_value = problem_type.get()

            genetic_algorithm(lower_limit, upper_limit, resolution, initial_pop_size, max_pop_size, num_generations,
                              mutation_prob_ind, mutation_prob_gen, problem_type_value, table)
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    tk.Button(root, text="Ejecutar", command=run_algorithm).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
