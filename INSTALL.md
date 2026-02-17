# Guía de Instalación - The Laughing Man Virtual Camera

Esta guía te llevará paso a paso por el proceso de instalación y configuración del filtro de cámara virtual.

## 📋 Requisitos Previos

- **Sistema Operativo**: Linux (Ubuntu 20.04+, Debian 11+, o similar)
- **Python**: 3.10 o superior
- **Webcam**: Compatible con V4L2
- **Permisos**: Acceso sudo para instalar paquetes del sistema

## 🔧 Instalación Paso a Paso

### 1. Actualizar el Sistema e Instalar Dependencias

```bash
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils libcairo2-dev
```

**¿Qué instala cada paquete?**
- `v4l2loopback-dkms`: Módulo del kernel para crear dispositivos de cámara virtual
- `v4l2loopback-utils`: Utilidades para gestionar v4l2loopback
- `libcairo2-dev`: Biblioteca necesaria para convertir SVG a PNG

### 2. Cargar el Módulo v4l2loopback

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1
```

**Parámetros explicados:**
- `devices=1`: Crear un solo dispositivo virtual
- `video_nr=10`: Número del dispositivo (`/dev/video10`)
- `card_label="Laughing-Man-Cam"`: Nombre que aparecerá en Google Meet
- `exclusive_caps=1`: Necesario para compatibilidad con navegadores modernos

**Verificar que el módulo se cargó correctamente:**
```bash
# Debería mostrar "v4l2loopback"
lsmod | grep v4l2loopback

# Debería mostrar "Laughing-Man-Cam" como /dev/video10
v4l2-ctl --list-devices
```

### 3. Configurar Permisos de Usuario

Añade tu usuario al grupo `video` para acceder a los dispositivos de cámara:

```bash
sudo usermod -aG video $USER
```

**⚠️ IMPORTANTE**: Debes cerrar sesión y volver a entrar para que los cambios surtan efecto.

Verifica que estás en el grupo:
```bash
groups | grep video
```

### 4. [OPCIONAL] Configurar Persistencia del Módulo

Si quieres que el módulo v4l2loopback se cargue automáticamente al iniciar el sistema:

```bash
# Cargar módulo al inicio
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf

# Configurar opciones del módulo
echo "options v4l2loopback devices=1 video_nr=10 card_label='Laughing-Man-Cam' exclusive_caps=1" | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

**Verificar configuración:**
```bash
cat /etc/modules-load.d/v4l2loopback.conf
cat /etc/modprobe.d/v4l2loopback.conf
```

### 5. Instalar Dependencias de Python

#### Opción A: Usando `uv` (⚡ RECOMENDADO)

`uv` es significativamente más rápido que pip tradicional y maneja mejor las dependencias.

```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Recargar shell para usar uv
source ~/.bashrc  # o ~/.zshrc si usas zsh

# Navegar al directorio del proyecto
cd /home/qnelo/develop/personal/thelaughing-man

# Crear entorno virtual
uv venv

# Activar entorno virtual
source .venv/bin/activate

# Instalar el proyecto y sus dependencias
uv pip install -e .
```

#### Opción B: Usando `venv` tradicional

```bash
# Navegar al directorio del proyecto
cd /home/qnelo/develop/personal/thelaughing-man

# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar el proyecto y sus dependencias
pip install -e .
```

**Verificar la instalación:**
```bash
python -c "import mediapipe, cv2, pyvirtualcam, cairosvg; print('✓ Todas las dependencias instaladas correctamente')"
```

## 🚀 Primera Ejecución

```bash
# Asegurarse de que el entorno virtual está activado
source .venv/bin/activate

# Ejecutar el script
python main.py
```

**Salida esperada:**
```
📥 Downloading logo from https://static.wikia.nocookie.net/...
✓ Logo downloaded successfully: assets/laughing_man.svg
🔄 Converting SVG to PNG...
✓ Logo converted to PNG: assets/laughing_man.png
📷 Initializing camera: /dev/video0...
✓ Camera initialized: 640x480 @ 30fps
🎥 Initializing virtual camera: /dev/video10...
✓ Virtual camera initialized: /dev/video10
🎭 Initializing face detection...
✓ Face detection initialized

============================================================
🎉 The Laughing Man Camera is now running!
============================================================
📹 Virtual camera is available at: /dev/video10
💡 In Google Meet, select 'Laughing-Man-Cam' as your camera
🛑 Press Ctrl+C to stop
============================================================

📊 FPS: 29.8
```

## 🎥 Configurar en Google Meet

1. Abre Google Meet en tu navegador preferido (Chrome/Firefox)
2. Inicia o únete a una reunión
3. Haz clic en los tres puntos (⋮) → **Configuración**
4. Ve a la pestaña **Video**
5. En "Cámara", selecciona **"Laughing-Man-Cam"**
6. ¡Deberías ver el filtro aplicado en tiempo real!

## 🔄 Uso Diario

### Iniciar la cámara virtual

```bash
cd /home/qnelo/develop/personal/thelaughing-man
source .venv/bin/activate
python main.py
```

### Detener la cámara virtual

Presiona `Ctrl+C` en la terminal donde se está ejecutando.

### Script de conveniencia (opcional)

Puedes crear un script para iniciar rápidamente:

```bash
#!/bin/bash
# Guardar como ~/start-laughing-man.sh

cd /home/qnelo/develop/personal/thelaughing-man
source .venv/bin/activate
python main.py
```

Hacerlo ejecutable:
```bash
chmod +x ~/start-laughing-man.sh
```

Ejecutar:
```bash
~/start-laughing-man.sh
```

## 🐛 Solución de Problemas

### El módulo v4l2loopback no se carga

**Error**: `modprobe: FATAL: Module v4l2loopback not found`

**Solución**:
```bash
# Reinstalar v4l2loopback-dkms
sudo apt remove v4l2loopback-dkms
sudo apt install v4l2loopback-dkms
```

### /dev/video10 no se crea

**Verificar dispositivos existentes**:
```bash
ls /dev/video*
```

Si `/dev/video10` ya existe, cambia el número:
```bash
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="Laughing-Man-Cam" exclusive_caps=1
```

Y actualiza `VIRTUAL_DEVICE` en `main.py`.

### Errores de permisos al acceder a /dev/videoX

**Error**: `Permission denied`

**Solución**:
```bash
# Verificar permisos
ls -l /dev/video*

# Añadir usuario al grupo video
sudo usermod -aG video $USER

# Cerrar sesión y volver a entrar
```

### Dependencias de Python no se instalan

**Error relacionado con `cairosvg`**:

Asegúrate de que `libcairo2-dev` está instalado:
```bash
sudo apt install -y libcairo2-dev pkg-config
```

**Error relacionado con `mediapipe`**:

MediaPipe puede requerir dependencias adicionales:
```bash
sudo apt install -y python3-dev build-essential
```

### La cámara no se detecta en Google Meet

1. **Verificar que el script está corriendo** y no hay errores
2. **Recargar la página** de Google Meet
3. **Verificar permisos de cámara** en el navegador
4. **Probar la cámara virtual** con otra herramienta primero:
   ```bash
   ffplay /dev/video10
   ```

## 📚 Recursos Adicionales

- [Documentación de v4l2loopback](https://github.com/umlaeute/v4l2loopback)
- [Documentación de MediaPipe](https://mediapipe.dev/)
- [Guía de pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [Documentación de uv](https://github.com/astral-sh/uv)

## ❓ ¿Necesitas Ayuda?

Si encuentras algún problema no cubierto en esta guía, abre un issue en el repositorio del proyecto.

---

¡Disfruta de tu cámara virtual con estilo! 🎭✨
