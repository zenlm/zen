/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  distDir: 'docs',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
  basePath: '',
}

module.exports = nextConfig
