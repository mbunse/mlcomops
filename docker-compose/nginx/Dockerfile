ARG NGINX_VERSION=1.21.3
ARG BITNAMI_NGINX_REVISION=r1
ARG BITNAMI_NGINX_TAG=${NGINX_VERSION}-debian-10-${BITNAMI_NGINX_REVISION}

FROM bitnami/nginx:${BITNAMI_NGINX_TAG} AS builder
USER root
# Redeclare NGINX_VERSION so it can be used as a parameter inside this build stage
ARG NGINX_VERSION
# Install required packages and build dependencies
RUN install_packages dirmngr gpg gpg-agent curl build-essential libpcre3-dev zlib1g-dev hgsubversion libssl-dev
# Add trusted NGINX PGP key for tarball integrity verification
#RUN gpg --keyserver pgp.mit.edu --recv-key 520A9993A1C052F8
# Download NGINX, verify integrity and extract
RUN cd /tmp && \
    curl -O http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz && \
    curl -O http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz.asc && \
    #gpg --verify nginx-${NGINX_VERSION}.tar.gz.asc nginx-${NGINX_VERSION}.tar.gz && \
    tar xzf nginx-${NGINX_VERSION}.tar.gz

RUN cd /tmp && \
    hg clone http://hg.nginx.org/njs
# Compile NGINX with desired module
RUN cd /tmp/nginx-${NGINX_VERSION} && \
    rm -rf /opt/bitnami/nginx && \
    ./configure --prefix=/opt/bitnami/nginx --with-compat --add-dynamic-module=../njs/nginx && \
    make && \
    make install

FROM bitnami/nginx:${BITNAMI_NGINX_TAG}
USER root
# Install ngx_http_perl_module system package dependencies
#RUN install_packages libperl-dev
# Install ngx_http_perl_module files
#COPY --from=builder /usr/local/lib/x86_64-linux-gnu/perl /usr/local/lib/x86_64-linux-gnu/perl
COPY --from=builder /opt/bitnami/nginx/modules/ngx_http_js_module.so /opt/bitnami/nginx/modules/ngx_http_js_module.so
# Enable module
RUN echo "load_module modules/ngx_http_js_module.so;" | cat - /opt/bitnami/nginx/conf/nginx.conf > /tmp/nginx.conf && \
    cp /tmp/nginx.conf /opt/bitnami/nginx/conf/nginx.conf
# Set the container to be run as a non-root user by default
USER 1001