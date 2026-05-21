import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
})

export const analyzeText = async (text) => {
  const response = await api.post('/api/detect', { text })
  return response.data
}

export const analyzeURL = async (url) => {
  const response = await api.post('/api/detect-url', { url })
  return response.data
}

export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}