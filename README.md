# Kids Care AI - Fall Detection and Cry Detection

Kids Care AI IOT Device - RaspberryPi USB Mic voice detection and Picamera fall detection.

## SSH Login to Raspberry PI

```cmd
ssh admin@raspberrypi.local
cd Documents/Kids_Care_AI
```
password: `admin`

---
## Run Fall Detection

 ```bash
 python fall.py 2>/dev/null
 ```

--- 

## Run Cry Detection

 ```bash
 python cry.py 2>/dev/null
 ```

### Test Mic

```bash
arecord -D hw:3,0 -f cd -c 1 -vv ~/test.wav
```

- hw:3 This is the card no(Hardware PCM card 3).
- -c 1: This sets the number of channels to 1 (mono).
- -f cd: This keeps the format as CD quality (16-bit, 44100 Hz).