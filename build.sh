#!/bin/bash
set -e

APP=darknet2onnx

echo "Building ${APP}..."

echo "linux/amd64"
export GOOS=linux && export GOARCH=amd64 && export CGO_ENABLED=0 && \
    go build -ldflags "-s -w" -o ${APP} -gcflags "all=-trimpath=$GOPATH" -trimpath . && \
    tar -czvf linux-amd64-${APP}.tar.gz ${APP}

echo "linux/arm64"
export GOOS=linux && export GOARCH=arm64 && export CGO_ENABLED=0 && \
    go build -ldflags "-s -w" -o ${APP} -gcflags "all=-trimpath=$GOPATH" -trimpath . && \
    tar -czvf linux-arm64-${APP}.tar.gz ${APP}

echo "windows/amd64"
export GOOS=windows && export GOARCH=amd64 && export CGO_ENABLED=0 && \
    go build -ldflags "-s -w" -o ${APP}.exe -gcflags "all=-trimpath=$GOPATH" -trimpath . && \
    zip windows-amd64-${APP}.zip ${APP}.exe

echo "darwin/amd64"
export GOOS=darwin && export GOARCH=amd64 && export CGO_ENABLED=0 && \
    go build -ldflags "-s -w" -o ${APP} -gcflags "all=-trimpath=$GOPATH" -trimpath . && \
    tar -czvf darwin-amd64-${APP}.tar.gz ${APP}

echo "darwin/arm64"
export GOOS=darwin && export GOARCH=arm64 && export CGO_ENABLED=0 && \
    go build -ldflags "-s -w" -o ${APP} -gcflags "all=-trimpath=$GOPATH" -trimpath . && \
    tar -czvf darwin-arm64-${APP}.tar.gz ${APP}

# сleanup
rm -f ${APP} ${APP}.exe

echo "Done!"
ls -lh *.tar.gz *.zip
