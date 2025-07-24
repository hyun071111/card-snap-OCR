# card-snap-OCR

### 환경설정
macOS 또는 Linux
```
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
또는 
brew install uv
```
windows
```aiignore
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
PyPI로 설치
```
# With pip.
pip install uv

# Or pipx.
pipx install uv
```

가상환경 설정
```bash
uv venv --python 3.11.0
```

requirements 설치
```bash
uv pip install requirements.txt
```



*참고 자료*   
- https://m.blog.naver.com/tommybee/221836206507
- https://m.blog.naver.com/tommybee/221837611962?recommendTrackingCode=2
- https://ahnanne.tistory.com/67
- https://velog.io/@joongwon00/인턴-프로젝트-2.-EasyOCR을-써보자