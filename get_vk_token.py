import requests

url = 'https://id.vk.com/oauth2/auth'
data = {
    'grant_type': 'authorization_code',
    'code': 'YOUR_CODE',  # Код из предыдущего шага
    'redirect_uri': 'https://oauth.vk.com/blank.html',
    'client_id': 'YOUR_ID',
    'device_id': 'YOUR_ID',  # Также из адресной строки после авторизации
    'code_verifier': 'YOUR_CODE'  # Сгенерированный ранее код
}
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

response = requests.post(url, data=data, headers=headers)
token_data = response.json()

# В token_data будет access_token и refresh_token
print(token_data)
