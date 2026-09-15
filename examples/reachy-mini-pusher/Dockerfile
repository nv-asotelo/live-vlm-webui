# Reachy Mini -> live-vlm-webui push client, with the GStreamer stack the SDK needs.
#
# Why a container. Talking to a *wireless* Reachy Mini from another machine puts the SDK on its
# WEBRTC media backend, which needs GStreamer >= 1.22 plus the `webrtcsrc` element. `webrtcsrc`
# lives in gst-plugins-rs and is not packaged for Linux at all, so it has to be compiled from Rust
# source. On an Ubuntu 22.04 host (GStreamer 1.20) the documented fix is a third-party multimedia
# PPA, which upgrades multimedia packages across the whole machine. Doing it here instead keeps the
# host untouched: Ubuntu 24.04 already ships GStreamer 1.24, and the Rust build is thrown away with
# the image.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Package list from the Reachy Mini SDK's Linux GStreamer instructions, plus the headers pip needs
# to build pygobject (Ubuntu ships 3.48, above the SDK's <=3.46 ceiling, so it gets built here).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      libgstreamer-plugins-bad1.0-dev \
      libglib2.0-dev \
      libssl-dev \
      libgirepository1.0-dev \
      libcairo2-dev \
      libportaudio2 \
      libnice10 \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-alsa \
      gstreamer1.0-nice \
      gstreamer1.0-tools \
      python3-gi \
      python3-gi-cairo \
      python3-dev \
      python3-venv \
      python3-pip \
      build-essential \
      pkg-config \
      git \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain, only to build the WebRTC plugin.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN cargo install cargo-c --locked

# 0.14.5 is pinned by the SDK docs: it carries a webrtcsink fix for a deadlock between remote
# description and ICE handling that older 0.14.x tags do not have.
RUN git clone --depth 1 --branch 0.14.5 \
      https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git /src/gst-plugins-rs \
 && cd /src/gst-plugins-rs \
 && cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release \
 && rm -rf /src /root/.cargo/registry /root/.cargo/git

ENV GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu

RUN python3 -m venv /venv \
 && /venv/bin/pip install --no-cache-dir --upgrade pip wheel \
 && /venv/bin/pip install --no-cache-dir reachy-mini opencv-python-headless requests

COPY push_reachy_mini.py /app/push_reachy_mini.py
WORKDIR /app

# Run with --network host: the client needs to reach the robot and the WebUI, and WebRTC negotiates
# arbitrary UDP ports that bridged networking would not forward.
ENTRYPOINT ["/venv/bin/python", "/app/push_reachy_mini.py"]
