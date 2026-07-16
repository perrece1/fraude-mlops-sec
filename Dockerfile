# 1. Usamos la imagen oficial ligera de Debian
FROM python:3.10-slim

# 2. Establecemos el directorio de trabajo
WORKDIR /app

# 3. Actualizamos el sistema para corregir las vulnerabilidades de Debian (glibc, util-linux, etc.)
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# 4. Copiamos el archivo de requisitos
COPY requirements.txt .

# 5. Instalamos las dependencias oficiales (ahora descargará las ruedas precompiladas .whl sin fallar)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiamos los archivos de la API y el modelo
COPY app.py .
COPY modelo_fraude.pkl .

# 7. Creamos y usamos un usuario sin privilegios para mitigar exploits del sistema operativo
RUN useradd -m appuser
USER appuser

# 8. Exponemos el puerto 8000
EXPOSE 8000

# 9. Ejecutamos la API sin --reload para producción
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]