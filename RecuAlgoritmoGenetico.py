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
def genetic_algorithm(lower_limit, upper_limit, resolution, initial_pop_size, max_pop_size, num_generations, mutation_prob_ind, mutation_prob_gen, problem_type, console_output, table):
    delta, bit_length, value_range = calculate_initial_params(lower_limit, upper_limit, resolution)
    population = generate_initial_population(initial_pop_size, value_range, bit_length, lower_limit, delta)

    generations = []
    best_y = []
    worst_y = []
    average_y = []

    # Graficar e imprimir la población inicial (Generación 0)
    console_output.insert(tk.END, "Generación 0:\n")
    plot_generation(0, population, lower_limit, upper_limit, problem_type)
    best_individual = max(population, key=lambda individual: individual['y']) if problem_type == "maximizacion" else min(population, key=lambda individual: individual['y'])
    for individual in population:
        console_output.insert(tk.END, f"id: {individual['id']}, decimal: {individual['decimal']}, x: {individual['x']:.3f}, y: {individual['y']:.3f}\n")
    table.insert("", tk.END, values=(0, best_individual['x'], best_individual['y']))

    for generation in range(1, num_generations + 1):
        pairs = pair_individuals(population)
        new_population = []

        # Elitismo: conservar el mejor individuo de la generación anterior
        best_individual_previous_generation = best_individual
        #---------------------------------------------- implementar en esta parte el promedio ----------------------------------------------
        new_population.append(best_individual_previous_generation)

        for parent1, parent2 in pairs:
            new_binary1, new_binary2 = multiple_point_crossover(parent1, parent2, bit_length)
            new_binary1 = mutate(new_binary1, mutation_prob_gen, mutation_prob_ind)
            new_binary2 = mutate(new_binary2, mutation_prob_gen, mutation_prob_ind)
            new_individual1, new_individual2 = create_new_individuals(new_binary1, new_binary2, lower_limit, delta)
            new_population.extend([new_individual1, new_individual2])

        # Poda de la población, manteniendo los mejores
        population = prune_population(new_population, max_pop_size, problem_type)

        # Graficar e imprimir la población de la generación actual
        console_output.insert(tk.END, f"Generación {generation}:\n")
        plot_generation(generation, population, lower_limit, upper_limit, problem_type)
        best_individual = max(population, key=lambda individual: individual['y']) if problem_type == "maximizacion" else min(population, key=lambda individual: individual['y'])
        worst_individual = min(population, key=lambda individual: individual['y']) if problem_type == "maximizacion" else max(population, key=lambda individual: individual['y'])
        average_individual = sum(individual['y'] for individual in population) / len(population)

        for individual in population:
            console_output.insert(tk.END, f"id: {individual['id']}, decimal: {individual['decimal']}, x: {individual['x']:.3f}, y: {individual['y']:.3f}\n")

        table.insert("", tk.END, values=(generation, best_individual['x'], best_individual['y']))

        generations.append(generation)
        best_y.append(best_individual['y'])
        worst_y.append(worst_individual['y'])
        average_y.append(average_individual)

    # Graficar estadísticas de todas las generaciones
    plot_statistics(generations, best_y, worst_y, average_y)

    # Crear el video de las generaciones en formato .mp4
    create_video_from_images('Generaciones', 'generaciones.mp4')

    console_output.insert(tk.END, "¡Ejecución del algoritmo completada!\n")
    console_output.yview(tk.END)

# Configuración de la interfaz gráfica
def setup_interface():
    root = tk.Tk()
    root.title("Algoritmo Genético")

    # Crear los widgets
    lbl_lower_limit = ttk.Label(root, text="Límite Inferior:")
    lbl_upper_limit = ttk.Label(root, text="Límite Superior:")
    lbl_resolution = ttk.Label(root, text="Resolución:")
    lbl_initial_pop_size = ttk.Label(root, text="Tamaño Población Inicial:")
    lbl_max_pop_size = ttk.Label(root, text="Tamaño Máximo Población:")
    lbl_num_generations = ttk.Label(root, text="Número de Generaciones:")
    lbl_mutation_prob_ind = ttk.Label(root, text="Probabilidad de Mutación (Individuo):")
    lbl_mutation_prob_gen = ttk.Label(root, text="Probabilidad de Mutación (Gen):")
    lbl_problem_type = ttk.Label(root, text="Tipo de Problema:")

    entry_lower_limit = ttk.Entry(root)
    entry_upper_limit = ttk.Entry(root)
    entry_resolution = ttk.Entry(root)
    entry_initial_pop_size = ttk.Entry(root)
    entry_max_pop_size = ttk.Entry(root)
    entry_num_generations = ttk.Entry(root)
    entry_mutation_prob_ind = ttk.Entry(root)
    entry_mutation_prob_gen = ttk.Entry(root)

    combo_problem_type = ttk.Combobox(root, values=["maximizacion", "minimizacion"])
    combo_problem_type.current(0)

    console_output = scrolledtext.ScrolledText(root, width=80, height=20)
    table = ttk.Treeview(root, columns=("Generación", "Mejor X", "Mejor Y"), show="headings")
    table.heading("Generación", text="Generación")
    table.heading("Mejor X", text="Mejor X")
    table.heading("Mejor Y", text="Mejor Y")

    # Crear botón para ejecutar el algoritmo
    btn_run = ttk.Button(root, text="Ejecutar", command=lambda: run_algorithm(entry_lower_limit, entry_upper_limit, entry_resolution, entry_initial_pop_size, entry_max_pop_size, entry_num_generations, entry_mutation_prob_ind, entry_mutation_prob_gen, combo_problem_type, console_output, table))

    # Colocar los widgets en el grid
    lbl_lower_limit.grid(row=0, column=0, sticky="e")
    lbl_upper_limit.grid(row=1, column=0, sticky="e")
    lbl_resolution.grid(row=2, column=0, sticky="e")
    lbl_initial_pop_size.grid(row=3, column=0, sticky="e")
    lbl_max_pop_size.grid(row=4, column=0, sticky="e")
    lbl_num_generations.grid(row=5, column=0, sticky="e")
    lbl_mutation_prob_ind.grid(row=6, column=0, sticky="e")
    lbl_mutation_prob_gen.grid(row=7, column=0, sticky="e")
    lbl_problem_type.grid(row=8, column=0, sticky="e")

    entry_lower_limit.grid(row=0, column=1)
    entry_upper_limit.grid(row=1, column=1)
    entry_resolution.grid(row=2, column=1)
    entry_initial_pop_size.grid(row=3, column=1)
    entry_max_pop_size.grid(row=4, column=1)
    entry_num_generations.grid(row=5, column=1)
    entry_mutation_prob_ind.grid(row=6, column=1)
    entry_mutation_prob_gen.grid(row=7, column=1)
    combo_problem_type.grid(row=8, column=1)

    console_output.grid(row=9, column=0, columnspan=2)
    table.grid(row=10, column=0, columnspan=2)
    btn_run.grid(row=11, column=0, columnspan=2)

    root.mainloop()

# Función para ejecutar el algoritmo con los parámetros de la interfaz
def run_algorithm(entry_lower_limit, entry_upper_limit, entry_resolution, entry_initial_pop_size, entry_max_pop_size, entry_num_generations, entry_mutation_prob_ind, entry_mutation_prob_gen, combo_problem_type, console_output, table):
    lower_limit = float(entry_lower_limit.get())
    upper_limit = float(entry_upper_limit.get())
    resolution = float(entry_resolution.get())
    initial_pop_size = int(entry_initial_pop_size.get())
    max_pop_size = int(entry_max_pop_size.get())
    num_generations = int(entry_num_generations.get())
    mutation_prob_ind = float(entry_mutation_prob_ind.get())
    mutation_prob_gen = float(entry_mutation_prob_gen.get())
    problem_type = combo_problem_type.get()

    genetic_algorithm(lower_limit, upper_limit, resolution, initial_pop_size, max_pop_size, num_generations, mutation_prob_ind, mutation_prob_gen, problem_type, console_output, table)

# Configurar y ejecutar la interfaz gráfica
setup_interface()
