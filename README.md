# The Laughing Man Virtual Camera 🎭

<div align="center">

![Ghost in the Shell](https://img.shields.io/badge/Inspired_by-Ghost_in_the_Shell-blueviolet)
![Python 3.10+](https://shields.io/badge/Python-3.10+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

*Superpón el icónico logo de "The Laughing Man" sobre tu rostro en tiempo real para videollamadas* 👤➡️🎭

![Demo](/home/qnelo/.gemini/antigravity/brain/a8d25e18-7047-452f-a350-b842d9273b45/demo_mockup.webp)

</div>

## 📖 Descripción

Este proyecto crea un filtro de cámara virtual que detecta tu rostro en tiempo real y superpone el logo rotatorio de "The Laughing Man" de Ghost in the Shell: Stand Alone Complex. La salida se envía a una cámara virtual compatible con Google Meet, Zoom y otras aplicaciones de videoconferencia.

## ✨ Características

- 🎯 **Detección de rostros** con MediaPipe (alta precisión y rendimiento)
- 🔄 **Rotación continua** del logo (fiel al anime)
- 🎨 **Alpha blending** para transparencia perfecta
- 📹 **Cámara virtual** compatible con aplicaciones de videollamadas
- ⚡ **Optimizado** para 30 FPS consistentes
- 🎭 **Descarga automática** del logo desde la fuente oficial

## 🔧 Requisitos del Sistema

- **OS**: Linux (Ubuntu/Debian recomendado)
- **Python**: 3.10 o superior
- **Hardware**: Webcam compatible con V4L2

### Dependencias del Sistema

```bash
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils libcairo2-dev
```

## 🚀 Instalación

Para instrucciones detalladas paso a paso, consulta [INSTALL.md](INSTALL.md).

### Instalación Rápida

```bash
# 1. Clonar el repositorio
cd /home/qnelo/develop/personal/thelaughing-man

# 2. Cargar el módulo v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1

# 3. Instalar con uv (recomendado)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. Ejecutar
python main.py
```

## 💻 Uso

1. **Iniciar la cámara virtual**:
   ```bash
   source .venv/bin/activate
   python main.py
   ```

2. **Configurar en Google Meet**:
   - Abre Google Meet en tu navegador
   - Ve a Configuración → Video
   - Selecciona **"Laughing-Man-Cam"** como tu cámara
   - ¡Listo! El filtro se aplicará automáticamente

3. **Detener la aplicación**:
   - Presiona `Ctrl+C` en la terminal

## 📊 Rendimiento

El sistema está optimizado para mantener 30 FPS constantes:
- ✅ Detección de rostros usando modelo ligero de MediaPipe
- ✅ Caché de logos redimensionados
- ✅ Alpha blending optimizado con NumPy
- ✅ Rotación eficiente con OpenCV

## 🔧 Configuración Avanzada

### Cambiar el dispositivo de cámara virtual

Edita `main.py` y modifica:
```python
VIRTUAL_DEVICE = "/dev/video10"  # Cambia el número si es necesario
```

### Ajustar la confianza de detección

En `main.py`, modifica:
```python
self.face_overlay = FaceOverlay(
    logo_path=str(LOGO_PNG_PATH),
    min_detection_confidence=0.5  # Rango: 0.0 - 1.0
)
```

## 🐛 Solución de Problemas

### Error: "Failed to open camera"
- Verifica que tu webcam esté conectada: `ls /dev/video*`
- Prueba con otro dispositivo: `CAMERA_DEVICE = "/dev/video1"`

### Error: "Failed to initialize virtual camera"
- Verifica que v4l2loopback esté cargado: `lsmod | grep v4l2loopback`
- Recarga el módulo: `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1`

### La detección de rostro es lenta
- Reduce la resolución de tu webcam
- Aumenta `min_detection_confidence` a 0.6 o 0.7

### Google Meet no muestra la cámara virtual
- Verifica que el dispositivo existe: `v4l2-ctl --list-devices`
- Reinicia el navegador después de iniciar el script
- Dale permisos de cámara al navegador

## 📄 Licencia

MIT License - Ver archivo LICENSE para más detalles

## 🙏 Créditos

- **Logo**: Ghost in the Shell: Stand Alone Complex
- **Face Detection**: [MediaPipe](https://mediapipe.dev/)
- **Virtual Camera**: [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- **Inspiración**: La icónica escena del "Laughing Man" 🎭

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

<div align="center">
Made with ❤️ and Python 🐍
</div>
