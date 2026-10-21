# Scripts

Solo se versionan aquí los scripts **reproducibles** del proyecto.

- `setup_env.sh`: crea el entorno conda y instala las dependencias.
- `cmds.sh`: chuleta de comandos de operación (GPU, procesos, Jupyter, TeX). No es un flujo automático.

Los lanzadores de una sola corrida —por ejemplo, un *watcher* que espera a que termine un proceso y arranca el siguiente— son **temporales**: viven en el servidor mientras dura la corrida y **no se commitean**. Si un flujo se vuelve recurrente, se convierte en un script reproducible y se documenta aquí.
