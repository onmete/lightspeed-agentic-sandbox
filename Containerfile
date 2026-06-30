# lightspeed-agentic-sandbox
#
# Multi-provider agent sandbox for OpenShift Lightspeed.
# Matches the production pod layout:
#   /app/skills  — skills mounted as OCI image volume
#   /tmp         — writable workspace for agent operations
#   /home/agent  — writable home directory
#
# Hermetic build: all dependencies are prefetched by Konflux/Hermeto.
# Network access is disabled during build.

# ---------------------------------------------------------------------------
# Builder stage: install Python deps from prefetched requirements
# ---------------------------------------------------------------------------
FROM registry.redhat.io/rhel9/python-312:latest AS builder

USER 0
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY requirements.*.txt ./

# Install Python packages from the platform-specific requirements file.
# In hermetic builds, Cachi2 sets PIP_* env vars pointing to prefetched deps.
# The unset avoids conflicts between Cachi2's --home and pip's --target.
RUN unset PIP_INSTALL_OPTIONS PIP_TARGET PIP_HOME PIP_PREFIX 2>/dev/null; \
    pip3.12 install --no-cache-dir --target /app/site-packages \
        -r requirements.$(uname -m).txt

# Install claude-code CLI.
# In hermetic builds (Konflux), Hermeto prefetches npm tarballs under
# /cachi2/output/deps/npm. Extract them directly — npm ci still reaches the
# registry when lockfile injection does not apply inside multi-stage COPYs.
# In non-hermetic builds (OpenShift CI), fall back to npm install -g.
COPY package.json package-lock.json ./
RUN dnf install -y --nodocs nodejs && dnf clean all
RUN if [ -f /cachi2/cachi2.env ]; then \
        VERSION=$(node -p "require('./package-lock.json').packages['node_modules/@anthropic-ai/claude-code'].version"); \
        case "$(uname -m)" in \
            x86_64) PLATFORM=linux-x64 ;; \
            aarch64) PLATFORM=linux-arm64 ;; \
            *) echo "unsupported arch: $(uname -m)"; exit 1 ;; \
        esac; \
        DEPS=/cachi2/output/deps/npm; \
        MAIN_TGZ=$(find "$DEPS" -name "*claude-code-${VERSION}.tgz" ! -name "*-${PLATFORM}*" ! -name "*darwin*" ! -name "*win32*" | head -1); \
        PLATFORM_TGZ=$(find "$DEPS" -name "*claude-code-${PLATFORM}-${VERSION}.tgz" | head -1); \
        test -n "$MAIN_TGZ" && test -n "$PLATFORM_TGZ"; \
        mkdir -p "node_modules/@anthropic-ai/claude-code" "node_modules/@anthropic-ai/claude-code-${PLATFORM}"; \
        tar -xzf "$MAIN_TGZ" -C "node_modules/@anthropic-ai/claude-code" --strip-components=1; \
        tar -xzf "$PLATFORM_TGZ" -C "node_modules/@anthropic-ai/claude-code-${PLATFORM}" --strip-components=1; \
    else \
        npm install -g @anthropic-ai/claude-code --ignore-scripts && \
        cp -a /usr/local/lib/node_modules /app/node_modules; \
    fi

# ---------------------------------------------------------------------------
# origincli stage: provides oc (kubectl is a symlink to oc in this image)
# ---------------------------------------------------------------------------
FROM registry.redhat.io/openshift4/ose-cli-rhel9:v4.21 AS origincli

# ---------------------------------------------------------------------------
# podman stage: provides catatonit init binary
# ---------------------------------------------------------------------------
FROM registry.redhat.io/ubi9/podman:9.8 AS podman

# ---------------------------------------------------------------------------
# Runtime stage: minimal image with only what the agent needs
# ---------------------------------------------------------------------------
FROM registry.redhat.io/rhel9/python-312-minimal:latest

USER 0
WORKDIR /app

# System packages (resolved from rpms.in.yaml via rpm prefetch).
# Split into functional groups for readability.

# Claude Code SDK requirements
RUN microdnf install -y --nodocs \
    bash git wget jq \
    && microdnf clean all

# SRE debugging toolkit
# tcpdump is installed separately with || true because it is not
# available in the UBI9 subset repos used by ci-operator builds (OpenShift CI).
# Konflux hermetic builds pre-fetch it via the RPM lockfile so the production
# image still includes tcpdump.
RUN microdnf install -y --nodocs \
    procps-ng iproute bind-utils net-tools openssl \
    lsof strace \
    less vim-minimal findutils file diffutils \
    skopeo unzip tar gzip \
    && (microdnf install -y --nodocs tcpdump || true) \
    && microdnf clean all

# Node.js runtime (for claude-code CLI)
RUN microdnf install -y --nodocs nodejs && microdnf clean all

# Copy Python site-packages from builder
COPY --from=builder /app/site-packages /opt/app-root/lib64/python3.12/site-packages

# Copy claude-code npm installation from builder
COPY --from=builder /app/node_modules /app/node_modules
RUN ln -s /app/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude

# oc from the origincli stage; kubectl is a symlink to oc in that image
COPY --from=origincli /usr/bin/oc /usr/bin/oc
RUN ln -s /usr/bin/oc /usr/bin/kubectl

# catatonit init binary from the podman stage
COPY --from=podman /usr/libexec/podman/catatonit /usr/bin/catatonit


# Copy application source and metadata
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml /app/README.md ./
COPY LICENSE /licenses/LICENSE

RUN mkdir -p /app/skills /tmp/agent-workspace /home/agent && \
    chown -R 1001:0 /app /home/agent /tmp/agent-workspace

ENV SHELL="/bin/bash"
ENV HOME="/home/agent"
ENV LIGHTSPEED_SKILLS_DIR="/app/skills"
ENV PYTHONPATH="/app/src:/opt/app-root/lib64/python3.12/site-packages"
ENV PATH="/usr/local/bin:${PATH}"

USER 1001:1001

EXPOSE 8080

ENTRYPOINT ["/usr/bin/catatonit", "--"]
CMD ["python3.12", "-m", "uvicorn", "lightspeed_agentic.app:app", "--host", "0.0.0.0", "--port", "8080"]

LABEL name="openshift-lightspeed/lightspeed-agentic-sandbox-rhel9" \
      summary="Multi-provider agent sandbox for OpenShift Lightspeed" \
      description="Python agent with Claude, Gemini, and OpenAI provider support" \
      cpe="cpe:/a:redhat:openshift_lightspeed:1::el9" \
      com.redhat.component="openshift-lightspeed" \
      io.k8s.display-name="OpenShift Lightspeed Agentic Sandbox" \
      io.k8s.description="Python agent with Claude, Gemini, and OpenAI provider support" \
      io.openshift.tags="openshift-lightspeed,ols" \
      konflux.additional-tags="latest"
