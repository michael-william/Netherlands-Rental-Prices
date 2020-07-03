mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"michaelcondonw@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
" > ~/.streamlit/config.toml