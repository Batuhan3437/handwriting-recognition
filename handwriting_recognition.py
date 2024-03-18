import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784')

#print(mnist.data.shape)
# Çıktı: (70000, 784)

# Verilere erişmek için kullanılan fonksiyon
def resmiGoster(dframe, indeks):
    some_digit = dframe.to_numpy()[indeks]
    some_digit_image = some_digit.reshape(28, 28)
    # Resmi oluşturma
    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")
    plt.show()

# Eğitim ve test verilerini bölelim
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# Test görüntülerinin bir kopyasını oluşturalım
test_img_kopya = test_img.copy()

# Test verisinin ilk elemanını göstermek için fonksiyonu kullanalım
#resmiGoster(test_img_kopya, 0)

# Ölçeklendirme yapmamız gerekiyor
scaler = StandardScaler()
train_img = scaler.fit_transform(train_img)
test_img = scaler.transform(test_img)

# PCA işlemini %95 varyansı koruyacak şekilde gerçekleştirelim
pca = PCA(.95)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# Varsayılan olarak lbfgs çözücüsü kullanılır, çünkü daha hızlıdır.
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=10000)
logisticRegr.fit(train_img, train_lbl)

while True:
    indeks = int(input("test_img'de hangi indeksi görmek istersiniz: "))
    # Test verisinin belirli bir indeksini göster
    if indeks >= 0:
        print(resmiGoster(test_img_kopya, indeks))
        print("\n\nİmagedeki sayının tahmini sonucu:", int(logisticRegr.predict(test_img[indeks].reshape(1, -1))))
        print("Evet için 1 \nHayır için 0 giriniz")
        deger=int(input("tekrar değer girmek istermisiniz: "))
        if deger==1:
            continue
        elif deger==0:
            break
        else:
            print("Geçersiz değer.tekrar deneyin")
            continue
    else:
        print("\n\nGeçersiz değer. Lütfen tekrar deneyin.\n\n")
        continue

    

# Yapay zekanın doğruluk oranını hesaplama
# logisticRegr.score(test_img, test_lbl)
# Çıktı: 0.9184, yani 9184 tanesini doğru bir şekilde sınıflandırmış, geri kalanları ise sınıflandıramamış.
