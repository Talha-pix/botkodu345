import discord
from discord.ext import commands
import os
import numpy as np  # NumPy
from PIL import Image, ImageOps  # Görüntü işlemleri
from tensorflow.keras.models import load_model  # Model yükleme
# Görsellerin kaydedileceği klasör
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

# Botu oluştur
bot = commands.Bot(command_prefix="!", intents=intents)

def get_class(image_path, model_path, labels_path):
    # Model ve etiket dosyalarının varlığını kontrol et
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return "Hata: Model veya etiket dosyası bulunamadı.", 0.0
    
    # Modeli yükle
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()
    
    
    # Görüntü ön işleme
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Tahmin yap
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'merhaba!Ben bir botum {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

# 'check' komutu tanımlıyoruz, bu komut görselleri kontrol edip kaydedecek
@bot.command()
async def check(ctx):
    # Mesajın ekli bir dosya içerip içermediğini kontrol ediyoruz
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:  # Mesajdaki tüm ekleri döngüyle kontrol ediyoruz
            file_name = attachment.filename  # Dosya adını alıyoruz
            

            # Dosyanın kaydedileceği yolu oluşturuyoruz
            file_path = os.path.join(IMAGE_DIR, file_name)
        try:
                await attachment.save(file_path)  # Dosyayı belirtilen klasöre kaydediyoruz
                await ctx.send(f"✅ Görsel başarıyla kaydedildi: `{file_path}`")  # Kullanıcıya başarı mesajı gönderiyoruz

                # Modeli çalıştır
                model_path = "keras_model.h5"
                labels_path = "labels.txt"
                class_name, confidence = get_class(file_path, model_path, labels_path)

                messages = {
    '0 cat':'Evcil kedi[1][4] (Felis catus[4] ya da Felis silvestris catus), küçük, genelde kıllı, evcilleştirilmiş, etobur memeli. Genelde ev hayvanı olarak beslenenlere ev kedisi,[5] ya da diğer kedigillerden ve küçük kedilerden ayırmak gerekmiyorsa kısaca kedi denir. İnsanlar kedilerin arkadaşlığına ve böcek gibi ev zararlılarını avlayabilme yeteneğine önem vermektedir.',
    '1 tiger':'Kaplan (Panthera tigris), kedigiller (Felidae) familyasından etçil bir memeli hayvan türü ve büyük kediler ailesinin dört üyesinden biridir. Panthera cinsinin en büyük kedisidir. Turuncu-kahverengi renge sahip kürkünün üzerindeki, koyu dikey çizgileri ile kolayca tanınabilir. Genellikle geyik ve yaban domuzu gibi toynaklıları avlayan bir süper avcıdır. Bölgeseldir ve genellikle yalnız ama sosyal bir avcıdır, yavruların yetiştirilmesi ve avlanmayı öğrenebilmesi için geniş bitişik yaşam alanlarına ihtiyaç duyarlar. Kaplan yavruları, bağımsız bir birey olmadan ve kendi yaşam alanlarını kurmak için ayrılmadan önce, anneleriyle yaklaşık iki yıl kalırlar.',
    '2 leopard': 'Leopar ( Panthera pardus ) , Panthera cinsindeki beş mevcut kedi türünden biridir . Koyu beneklerin rozetler halinde toplandığı soluk sarımsı ila koyu altın rengi bir kürkü vardır . Vücudu ince ve kaslıdır, 92-183 cm (36-72 inç) uzunluğa , 66-102 cm (26-40 inç) uzunluğunda bir kuyruğa ve 60-70 cm (24-28 inç) omuz yüksekliğine ulaşır. Erkekler genellikle 30,9-72 kg (68-159 lb) ve dişiler 20,5-43 kg (45-95 lb) ağırlığındadır.',
    '3 lion' : 'Aslan (Panthera leo), Afrika ve Hindistan a özgü Panthera cinsinden büyük bir kedidir. Kaslı, geniş göğüslü bir gövdesi; kısa, yuvarlak bir kafası; yuvarlak kulakları ve kuyruğunun ucunda koyu, tüylü bir tutamı vardır. Cinsel olarak dimorfiktir; yetişkin erkek aslanlar dişilerden daha büyüktür ve belirgin bir yeleleri vardır. Gurur adı verilen gruplar oluşturan sosyal bir türdür. Bir aslan grubu birkaç yetişkin erkek, akraba dişiler ve yavrulardan oluşur. Dişi aslan grupları genellikle birlikte avlanır ve çoğunlukla orta ve büyük toynaklı hayvanları avlar. Aslan bir tepe ve temel avcıdır.',
        }
    
                special_message = messages.get(class_name, "Bu sınıf için özel bir mesaj yok.")
            
                await ctx.send(f"🔍 Tahmin: `{class_name[2:]}` (%{confidence*100:.2f} güven) {special_message}")

        except Exception as e:  # Eğer bir hata oluşursa
                await ctx.send(f"⚠️ Görsel kaydedilirken hata oluştu: {str(e)}")  # Kullanıcıya hata mesajı gönderiyoruz
    else:
        await ctx.send("⚠️ Görsel yüklemeyi unuttun!")  # Kullanıcıya görsel yüklemesi gerektiğini hatırlatıyoruz

bot.run("")

