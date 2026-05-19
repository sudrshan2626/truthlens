import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
})

export const analyzeText = async (text) => {
  const response = await api.post('/api/detect', { text })
  return response.data
}

// NEW
export const analyzeURL = async (url) => {
  const response = await api.post('/api/detect-url', { url })
  return response.data
}

export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}