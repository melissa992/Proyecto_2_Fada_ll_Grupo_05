import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pulp
import os
import re
import time  # Para medir el tiempo

def cargar_datos(archivo):
    with open(archivo, 'r') as f:
        lineas = [line.strip() for line in f if line.strip() != '']
        n = int(lineas[0])
        m = int(lineas[1])
        p = list(map(int, lineas[2].split(',')))
        ext = list(map(float, lineas[3].split(',')))
        cei = list(map(float, lineas[4].split(',')))
        c = [list(map(float, lineas[5 + i].split(','))) for i in range(m)]
        ct = float(lineas[5 + m])
        maxM = int(lineas[6 + m])
    return n, m, p, ext, cei, c, ct, maxM

def guardar_datos_dzn(datos, ruta_dzn='DatosProyecto.dzn'):
    n, m, p, ext, cei, c, ct, maxM = datos
    with open(ruta_dzn, 'w') as f:
        f.write(f"n = {n};\n")
        f.write(f"m = {m};\n")
        f.write(f"p = [{','.join(map(str, p))}];\n")
        f.write(f"v = [{','.join(map(str, ext))}];\n")
        f.write(f"ce = [{','.join(map(str, cei))}];\n")
        matriz_c = ",\n".join("[" + ",".join(map(str, fila)) + "]" for fila in c)
        f.write(f"c = [|{matriz_c}|];\n")
        f.write(f"ct = {ct};\n")
        f.write(f"maxM = {maxM};\n")

def ejecutar_modelo_minizinc(modelo_path="Proyecto.mzn", dzn_path="DatosProyecto.dzn"):
    if not os.path.isfile(modelo_path):
        return f"❌ No se encontró el archivo MiniZinc: {modelo_path}"
    if not os.path.isfile(dzn_path):
        return f"❌ No se encontró el archivo de datos DZN: {dzn_path}"

    try:
        resultado = subprocess.run(
            ["minizinc", "--solver", "Gecode", "--search-complete-msg", "✓ Completed", modelo_path, dzn_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        if resultado.returncode != 0:
            return f"⚠️ ejecutado:\n"
        return resultado.stdout
    except Exception as e:
        return f"❌ MiniZinc: {str(e)}"

def mostrar_datos_en_interfaz(datos, salida_texto):
    n, m, p, ext, cei, c, ct, maxM = datos
    salida_texto.delete("1.0", tk.END)
    salida_texto.insert(tk.END, f" Datos del archivo seleccionado:\n\n")
    salida_texto.insert(tk.END, f" Total de personas (n): {n}\n")
    salida_texto.insert(tk.END, f" Número de opiniones (m): {m}\n")
    salida_texto.insert(tk.END, f" Distribución inicial p: {p}\n")
    salida_texto.insert(tk.END, f" Nivel de extremismo ext: {ext}\n")
    salida_texto.insert(tk.END, f" Costos extra cei: {cei}\n")
    salida_texto.insert(tk.END, f" Matriz de costos c:\n")
    for fila in c:
        salida_texto.insert(tk.END, f"   {fila}\n")
    salida_texto.insert(tk.END, f"\n Costo total máximo permitido: {ct}\n")
    salida_texto.insert(tk.END, f" Máx. número de movimientos: {maxM}\n\n")

def resolver_minext(datos, salida_texto):
    n, m, p, ext, cei, c, ct, maxM = datos
    model = pulp.LpProblem("MinExt", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(m)), lowBound=0, cat='Integer')

    final_p_expr = [
        p[j] + pulp.lpSum([x[i, j] for i in range(m) if i != j]) - pulp.lpSum([x[j, k] for k in range(m) if k != j])
        for j in range(m)
    ]

    model += pulp.lpSum([final_p_expr[j] * ext[j] for j in range(m)]), "TotalExtremismo"

    for i in range(m):
        model += pulp.lpSum([x[i, j] for j in range(m)]) <= p[i]

    total_cost_expr = []
    for i in range(m):
        for j in range(m):
            if i != j:
                base = c[i][j] * (1 + p[i]/n)
                extra = cei[j] if p[j] == 0 else 0
                total_cost_expr.append(x[i, j] * (base + extra))

    model += pulp.lpSum(total_cost_expr) <= ct
    model += pulp.lpSum([x[i, j] * abs(j - i) for i in range(m) for j in range(m)]) <= maxM

    # Medir tiempo de resolución
    start_time = time.time()
    model.solve()
    end_time = time.time()
    tiempo = end_time - start_time

    salida_texto.insert(tk.END, "\n✅ RESULTADOS:\n")
    salida_texto.insert(tk.END, f"Estado de la solución: {pulp.LpStatus[model.status]}\n")
    salida_texto.insert(tk.END, f"Extremismo mínimo alcanzado: {pulp.value(model.objective):.4f}\n\n")
    salida_texto.insert(tk.END, "📋 Movimientos sugeridos:\n")

    movimientos = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            val = pulp.value(x[i, j])
            if val and val > 0:
                salida_texto.insert(tk.END, f"   Mover {int(val)} persona(s) de opinión {i+1} a {j+1}\n")
                movimientos[i, j] = int(val)

    final_p_values = [pulp.value(expr) for expr in final_p_expr]

    # Cálculos adicionales
    extremismo_inicial = sum(p[j] * ext[j] for j in range(m))
    extremismo_final = pulp.value(model.objective)
    costo_inicial = 0  # Antes de movimientos no hay costo
    costo_final = pulp.value(pulp.lpSum(total_cost_expr))
    movimientos_inicial = maxM
    movimientos_final = sum(pulp.value(x[i, j]) * abs(j - i) for i in range(m) for j in range(m))

    salida_texto.insert(tk.END, "\n📊 Resumen Final:\n")
    salida_texto.insert(tk.END, f"Extremismo Inicial: {extremismo_inicial:.4f}\n")
    salida_texto.insert(tk.END, f"Extremismo Final: {extremismo_final:.4f}\n")
    salida_texto.insert(tk.END, f"Costo Inicial: {costo_inicial:.4f}\n")
    salida_texto.insert(tk.END, f"Costo Final: {costo_final:.4f}\n")
    salida_texto.insert(tk.END, f"Movimientos Inicial (máx): {movimientos_inicial}\n")
    salida_texto.insert(tk.END, f"Movimientos Final (usados): {movimientos_final:.2f}\n")
    salida_texto.insert(tk.END, f"Tiempo de resolución (s): {tiempo:.4f}\n")

    generar_graficos(p, final_p_values, movimientos, extremismo_final, costo_final, m)

def generar_graficos(p_inicial, p_final, movimientos, extremismo_final, costo_total_final, m):
    opiniones = [f"Op. {i+1}" for i in range(m)]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ancho_barra = 0.35
    indices = np.arange(len(opiniones))

    ax1.bar(indices - ancho_barra/2, p_inicial, ancho_barra, label='Inicial')
    ax1.bar(indices + ancho_barra/2, p_final, ancho_barra, label='Final')

    ax1.set_xlabel('Opinión')
    ax1.set_ylabel('Número de Personas')
    ax1.set_title('Distribución de Personas por Opinión (Inicial vs. Final)')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(opiniones)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    cax = ax2.matshow(movimientos, cmap='Blues')
    fig2.colorbar(cax, label='Número de Personas Movidas')

    for (i, j), val in np.ndenumerate(movimientos):
        if val > 0:
            ax2.text(j, i, int(val), va='center', ha='center', color='black', fontsize=10)

    ax2.set_xticks(np.arange(m))
    ax2.set_yticks(np.arange(m))
    ax2.set_xticklabels(opiniones)
    ax2.set_yticklabels(opiniones)
    ax2.set_xlabel('Opinión Destino')
    ax2.set_ylabel('Opinión Origen')
    ax2.set_title('Matriz de Movimientos de Personas entre Opiniones')
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    metricas = ['Extremismo Final', 'Costo Total']
    valores = [extremismo_final, costo_total_final]
    colores = ['skyblue', 'lightcoral']

    ax3.bar(metricas, valores, color=colores)
    ax3.set_ylabel('Valor')
    ax3.set_title('Resultados Clave del Modelo')
    for i, v in enumerate(valores):
        ax3.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def iniciar_gui():
    datos_leidos = {"valores": None}
    numero_archivo_actual = None

    ventana = tk.Tk()
    ventana.title("MinExt - Minimizar Extremismo")
    ventana.geometry("900x650")

    tk.Label(ventana, text="Selecciona archivo de entrada (.txt)", font=("Arial", 12)).pack(pady=10)

    salida_texto = scrolledtext.ScrolledText(ventana, width=120, height=30)
    salida_texto.pack(padx=10, pady=10)

    def seleccionar_archivo():
        nonlocal numero_archivo_actual
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            try:
                datos = cargar_datos(ruta)
                datos_leidos["valores"] = datos
                mostrar_datos_en_interfaz(datos, salida_texto)
                nombre = os.path.basename(ruta)
                encontrado = re.search(r'(\d+)', nombre)
                if encontrado:
                    numero_archivo_actual = int(encontrado.group(1))
                else:
                    numero_archivo_actual = 1
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo leer el archivo:\n{str(e)}")

    def ejecutar_modelo():
        if datos_leidos["valores"]:
            resolver_minext(datos_leidos["valores"], salida_texto)
        else:
            messagebox.showwarning("Advertencia", "Primero selecciona y carga un archivo válido.")

    def ejecutar_con_minizinc():
        if datos_leidos["valores"]:
            num = numero_archivo_actual if numero_archivo_actual is not None else 1
            nombre_dzn = f"DatosProyecto_{num}.dzn"
            guardar_datos_dzn(datos_leidos["valores"], ruta_dzn=nombre_dzn)
            salida_texto.insert(tk.END, f"\n✅ Archivo '{nombre_dzn}' creado correctamente.\n")
            salida_texto.see(tk.END)
            salida = ejecutar_modelo_minizinc(dzn_path=nombre_dzn)
            salida_texto.insert(tk.END, "\n📤 Resultado MiniZinc:\n")
            salida_texto.insert(tk.END, salida + "\n")
            salida_texto.see(tk.END)
        else:
            messagebox.showwarning("Advertencia", "Primero carga un archivo válido.")

    tk.Button(ventana, text="📂 Seleccionar archivo", command=seleccionar_archivo).pack()
    tk.Button(ventana, text="▶ Ver resultados", command=ejecutar_modelo, bg="green", fg="white").pack(pady=5)
    tk.Button(ventana, text="🚀 Crear DatosProyecto.dzn", command=ejecutar_con_minizinc, bg="blue", fg="white").pack(pady=5)

    ventana.mainloop()

if __name__ == "__main__":
    iniciar_gui()
