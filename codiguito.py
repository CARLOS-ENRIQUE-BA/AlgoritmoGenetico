# configuracion.py
import tkinter as tk

# Definir configuracion fuera de la función obtener_configuracion()
configuracion = {}

def obtener_configuracion():
    ventana = tk.Tk()
    ventana.title("Configuración del Algoritmo Genético")

    # Cambiar el color de fondo de la ventana
    ventana.config(bg="#333333")

    # Función para manejar el botón de "Aceptar"
    def aceptar_configuracion():
        # Utilizar global para indicar que se está modificando la variable configuracion global
        global configuracion
        configuracion["A"] = int(entrada_A.get())
        configuracion["B"] = int(entrada_B.get())
        configuracion["delta_x"] = float(entrada_delta_x.get())
        configuracion["num_generaciones"] = int(entrada_num_generaciones.get())
        configuracion["prob_mutacion"] = float(entrada_prob_mutacion.get())
        configuracion["cant_padres"] = int(entrada_cant_padres.get())
        configuracion["poblacion_maxima"] = int(entrada_poblacion_maxima.get())
        configuracion["maximizar"] = maximizar_var.get()
        ventana.destroy()

    # Crear los elementos de la interfaz gráfica
    tk.Label(ventana, text="Valor inicial (A):", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=0, column=0, padx=5, pady=5)
    entrada_A = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_A.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(ventana, text="Valor final (B):", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=1, column=0, padx=5, pady=5)
    entrada_B = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_B.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(ventana, text="Valor de delta x:", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=2, column=0, padx=5, pady=5)
    entrada_delta_x = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_delta_x.grid(row=2, column=1, padx=5, pady=5)

    tk.Label(ventana, text="Número de generaciones:", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=3, column=0, padx=5, pady=5)
    entrada_num_generaciones = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_num_generaciones.grid(row=3, column=1, padx=5, pady=5)

    tk.Label(ventana, text="Probabilidad de mutación (entre 0 y 1):", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=4, column=0, padx=5, pady=5)
    entrada_prob_mutacion = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_prob_mutacion.grid(row=4, column=1, padx=5, pady=5)

    tk.Label(ventana, text="Cantidad de padres:", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=5, column=0, padx=5, pady=5)
    entrada_cant_padres = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_cant_padres.grid(row=5, column=1, padx=5, pady=5)

    tk.Label(ventana, text="Población máxima:", font=("Helvetica", 14), bg="#333333", fg="white").grid(row=6, column=0, padx=5, pady=5)
    entrada_poblacion_maxima = tk.Entry(ventana, font=("Helvetica", 14))
    entrada_poblacion_maxima.grid(row=6, column=1, padx=5, pady=5)

    maximizar_var = tk.BooleanVar()
    maximizar_var.set(True)
    # Cambiar el color del borde del Checkbutton y desactivar el efecto de hover
    tk.Checkbutton(ventana, text="Maximizar", font=("Helvetica", 14), variable=maximizar_var, bg="#333333", fg="white", selectcolor="#444444", highlightthickness=0, activebackground="#333333", activeforeground="white").grid(row=7, column=0, columnspan=2, pady=10)

    boton_aceptar = tk.Button(ventana, text="Aceptar", font=("Helvetica", 14), command=aceptar_configuracion, bg="#555555", fg="white", relief=tk.FLAT)  # Quitar el efecto de relieve del botón
    boton_aceptar.grid(row=8, column=0, columnspan=2, pady=15)

    ventana.mainloop()

    return configuracion




if __name__ == "__main__":
    configuracion = obtener_configuracion()
    print("Configuración ingresada:")
    print(configuracion)