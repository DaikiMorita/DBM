import requests
import json

requests.post('上でコピったURL', data=json.dumps({
    'text': u'Test',  # 投稿するテキスト
    'username': u'me',  # 投稿のユーザー名
    'icon_emoji': u':ghost:',  # 投稿のプロフィール画像に入れる絵文字
    'link_names': 1,  # メンションを有効にする
}))
