# Incidencia: GPU no disponible en el servidor IIMAS

**Fecha:** 2026-11-03
**Host:** `helena-Precision-7920-Tower` (`ssh iimas`, 132.248.51.54)
**Usuario que reporta:** `jmendoza`

## Síntoma

`nvidia-smi` falla con:

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

Las dos GPU siguen visibles en el bus PCI:

```
0000:17:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A5000]
0000:73:00.0 VGA compatible controller: NVIDIA Corporation GA102GL [RTX A5000]
```

pero no existen `/dev/nvidia*`, el módulo `nvidia` no está cargado (`lsmod` no
lo muestra) y `/proc/driver/nvidia/version` no existe. `lspci -k` solo ofrece
`nvidiafb` y `nouveau` como módulos posibles.

## Causa

El servidor arrancó con el kernel **6.8.0-138-generic**. El módulo DKMS de
NVIDIA está compilado únicamente para **6.8.0-40-generic**:

```
$ dkms status
nvidia/535.154.05, 6.8.0-40-generic, x86_64: installed
```

El kernel 6.8.0-138 se instaló el 2026-11-03 y no tiene build de DKMS:

```
$ ls /lib/modules/6.8.0-138-generic/updates/dkms
ls: cannot access ...: No such file or directory
```

Los headers del kernel 138 sí están instalados
(`/usr/src/linux-headers-6.8.0-138-generic`), Secure Boot está desactivado y
el kernel anterior 6.8.0-40 sigue instalado (`/boot/vmlinuz-6.8.0-40-generic`).
Por eso el problema es la falta de reconstrucción del módulo, no el hardware.

## Impacto

Sin GPU no se pueden rehacer las corridas de los reordenadores
(`bge-reranker-v2-m3`, `jina-reranker-v3`, `jina-colbert-v2`) sobre las 170 055
consultas de prueba. La alternativa en CPU no es viable: las corridas actuales,
con contextos mucho más cortos, tardaron entre 23 y 60 horas en dos A5000.

## Acciones solicitadas (requieren `sudo`)

Una de las dos:

1. **Reconstruir DKMS para el kernel en uso** y recargar el módulo:

   ```bash
   sudo dkms autoinstall -k 6.8.0-138-generic
   sudo modprobe nvidia
   nvidia-smi
   ```

   Si `dkms autoinstall` no encuentra las fuentes, reinstalar el paquete:

   ```bash
   sudo apt install --reinstall nvidia-dkms-535
   sudo dkms autoinstall -k 6.8.0-138-generic
   ```

2. **Arrancar el kernel 6.8.0-40**, que ya tiene el módulo compilado, y fijarlo
   temporalmente en GRUB hasta reconstruir DKMS para 138.

## Verificación

Después de la corrección, `nvidia-smi` debe listar las dos A5000:

```
$ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
index, name, memory.total [MiB], driver_version
0, NVIDIA RTX A5000, 24564 MiB, 535.154.05
1, NVIDIA RTX A5000, 24564 MiB, 535.154.05
```

También debe existir `/dev/nvidia0`, `/dev/nvidia1` y
`/proc/driver/nvidia/version`.

## Nota para el reporte

No se modificó ningún archivo del sistema ni se intentó cargar el módulo: el
usuario `jmendoza` no pertenece al grupo `sudo` (`groups` muestra
`jmendoza, alumnos, usbusers`), así que la corrección queda en manos de un
administrador del servidor.
