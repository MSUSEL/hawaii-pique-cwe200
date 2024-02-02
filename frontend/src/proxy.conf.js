const PROXY_CONFIG = [
  {
    context: [
      "/api",
    ],
    target: "https://localhost:44342",
    changeOrigin: true,
    secure: false,
  }
]

module.exports = PROXY_CONFIG;
