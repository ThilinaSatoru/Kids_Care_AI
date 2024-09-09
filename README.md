# Kids Care AI - Fall Detection and Cry Detection

Kids Care AI IOT Device - RaspberryPi USB Mic voice detection and Picamera fall detection.

## SSH Login to Raspberry PI

```cmd
ssh admin@raspberrypi.local
```

`password: admin`

## Test Picam

```cmd
python Documents/cam/stream.py
```
## Run Fall Detection
 ```bash
 python Documents/Kids_Care_AI/fall.py
 ```
## Run Cry Detection
 ```bash
 python Documents/Kids_Care_AI/cry.py
 ```

---

# Debugging
### Test Mic

```bash
arecord -D hw:3,0 -f cd -c 1 -vv ~/test.wav
```

- hw:3 This is the card no(Hardware PCM card 3).
- -c 1: This sets the number of channels to 1 (mono).
- -f cd: This keeps the format as CD quality (16-bit, 44100 Hz).

# Shutdown

```bash
sudo shutdown now
```