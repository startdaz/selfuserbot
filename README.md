[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://www.heroku.com/deploy?template=https://github.com/peerids/selfbot)

#### Deploy with Docker

#
> Clone Repository
```bash
git clone https://github.com/peerids/selfbot
```

#
> Install Docker (Skip if Already Installed)
```bash
sudo apt update && sudo apt install -y docker.io
```

#
> Build Docker Image
```bash
docker build ./selfbot -t selfbot
```

#
> More Session String: Separates with Spaces
```bash
docker run -d --name selfbot \
  -e "SESSION_STRING=STRING_1 STRING_2 etc..." \
  -e "BOT_TOKEN=123:Abc" \
  selfbot
```

#
> Get Container Logs
```bash
docker logs -f selfbot
```
