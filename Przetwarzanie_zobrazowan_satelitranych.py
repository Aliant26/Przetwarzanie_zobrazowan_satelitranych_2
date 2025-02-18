import streamlit as st
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

#włączenie pełnej szerokosci strony
st.set_page_config(layout="wide")

#komunikat
st.subheader("Przetwarzanie zobrazowań :blue[satelitarnych] :satellite:")
st.write("""Strona umożliwia proste przetwarzanie zobrazowań satelitarnych :star2: Niestety obrazy mogą zajmować maksymalnie 200 mb :pensive:""")

#definiowanie metod filtracji

prewitt_pionowy = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])


laplasjan = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])


sobel_poziomy = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])


usredniajacy = np.ones((3, 3), dtype=np.float32) / 9


# Wczytanie pliku
plik = st.file_uploader("Wczytaj zdjęcie", type=["tif", "jpg", "png"])

if plik is not None:
    st.success("Zdjęcie zostało załadowane poprawnie!")


    with rasterio.open(plik) as src:
        liczba_pasm = src.count
        pasma = {f"Kanał {i}": i for i in range(1, liczba_pasm + 1)}
        wybrane_pasmo = st.selectbox("Wybierz kanał do przetwarzania:", list(pasma.keys()))
        img = src.read(pasma[wybrane_pasmo])

        #zmiana do zakresu 0-255
        if img.dtype == np.uint16:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    obraz = img.copy()

    def resize_image(image, width=400, height=400):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = resize_image(img)
    obraz = resize_image(obraz)

#checkboxy do wyboru rodzaju filtracji
    st.write(""":blue[Jaki rodzaj filtracji chcesz zastosować?
    ***Zaznacz tylko jedną opcję!***]""")
    low_pass = st.checkbox("Filtracja dolnoprzepustowa")
    high_pass = st.checkbox("Filtracja górnoprzepustowa")

    metoda_low = None
    metoda_high = None
    przetworzony_obraz = None

#zdefiniowanie metody detekcji
    if low_pass:
        metoda_low = st.selectbox(
            "Wybierz filtr dolnoprzepustowy:",
            ("Gaussa", "Uśredniający"),
            index=None,
            placeholder="Filtr..."
        )

    if high_pass:
        metoda_high = st.selectbox(
            "Wybierz filtr górnoprzepustowy:",
            ("Sobela poziomy", "Prewitta pionowy", "Laplasjanowy"),
            index=None,
            placeholder="Filtr.."
        )

    #przetwarzanie obrazu
    if st.button("Dokonaj filtracji"):
        progress_bar = st.progress(0)

        for procent in range(100):
            time.sleep(0.1)
            progress_bar.progress(procent + 1)

        if metoda_low:
            if metoda_low == "Gaussa":
                przetworzony_obraz = cv2.GaussianBlur(obraz, (5, 5), 0)
            elif metoda_low == "Uśredniający":
                przetworzony_obraz = cv2.filter2D(src=obraz, ddepth=-1, kernel=usredniajacy)

            st.write("***Wynik filtracji dolnoprzepustowej***")

        elif metoda_high:
            if metoda_high == "Sobela poziomy":
                przetworzony_obraz = convolve2d(obraz, sobel_poziomy, mode = 'same', boundary= 'symm')
            elif metoda_high == "Prewitta pionowy":
                przetworzony_obraz = convolve2d(obraz, prewitt_pionowy, mode = 'same', boundary= 'symm')
            elif metoda_high == "Laplasjanowy":
                przetworzony_obraz = convolve2d(obraz, laplasjan, mode = 'same', boundary= 'symm')

            st.write("***Wynik filtracji górnoprzepustowej***")

        else:
            st.warning("Wybierz filtr przed przetwarzaniem!")
            przetworzony_obraz = None

        progress_bar.progress(100)
        #kolumny
        col1, col2 = st.columns(2)
        col1.image(img_resized, caption="Wybrany kanał bez przetworzenia")
        if przetworzony_obraz is not None:
            wynik = cv2.convertScaleAbs(przetworzony_obraz)
            col2.image(resize_image(wynik), caption="Wybrany kanał po filtracji")
        else:
            st.warning("Brak przetworzonego obrazu do wyświetlenia!")

#ocena strony
st.write("---")
st.write(":blue[Podziel się swoją opinią o tej stronie!]")
sentiment_mapping = ["1", "2", "3", "4", "5"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"Twoja ocena to: {sentiment_mapping[selected]} :star2:")
