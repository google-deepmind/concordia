# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Goods definitions for the Marketplace experiments."""

ORIGINAL_GOODS = {
    "Food": {
        "Low": {
            "Street Taco": {
                "price": 2.0,
                "inventory": 100,
                "advert": (
                    "Authentic flavor, unbeatable price! Perfectly seasoned,"
                    " grilled to order, and served on a warm corn tortilla. The"
                    " true taste of the street."
                ),
            },
            "Cup Noodles": {
                "price": 2.5,
                "inventory": 100,
                "advert": (
                    "Quick, easy, and delicious. Your perfect savory meal is"
                    " just three minutes away. Just add hot water and enjoy."
                ),
            },
        },
        "Mid": {
            "Gourmet Burger": {
                "price": 15.0,
                "inventory": 50,
                "advert": (
                    "A burger experience like no other. A juicy, thick-cut"
                    " patty made with premium, locally-sourced beef, served on"
                    " a toasted brioche bun."
                ),
            },
            "Sushi Platter": {
                "price": 22.0,
                "inventory": 40,
                "advert": (
                    "A fresh and vibrant selection of our finest rolls and"
                    " nigiri. A taste of Japan in every artfully prepared bite."
                ),
            },
        },
        "High": {
            "Michelin Star Meal": {
                "price": 75.0,
                "inventory": 10,
                "advert": (
                    "An unforgettable culinary journey. Allow our celebrated"
                    " chef to present a multi-course tasting menu that will"
                    " delight your senses."
                ),
            },
            "Omakase Experience": {
                "price": 90.0,
                "inventory": 8,
                "advert": (
                    "Trust the chef. An intimate, curated selection of the"
                    " finest seasonal fish and delicacies, flown in daily and"
                    " prepared before your eyes."
                ),
            },
        },
    },
    "Clothing": {
        "Low": {
            "Brown Fast Fashion T-Shirt": {
                "price": 5.0,
                "inventory": 100,
                "advert": (
                    "Trendy, affordable, and effortlessly cool. Update your"
                    " style for any season without breaking the bank."
                ),
            },
            "White Cotton T-Shirt": {
                "price": 6.0,
                "inventory": 100,
                "advert": (
                    "The essential classic for any wardrobe. Made from 100%"
                    " soft, breathable cotton for a perfect, comfortable fit"
                    " every time."
                ),
            },
        },
        "Mid": {
            "Levi's Jeans": {
                "price": 80.0,
                "inventory": 50,
                "advert": (
                    "Quality never goes out of style. The original, iconic blue"
                    " jean, crafted from durable denim for a timeless look and"
                    " feel."
                ),
            },
            "AllSaints Leather Jacket": {
                "price": 100.0,
                "inventory": 25,
                "advert": (
                    "Define your edge. Cut from supple, premium lambskin"
                    " leather with signature metal hardware. A timeless"
                    " investment piece."
                ),
            },
        },
        "High": {
            "Armani Suit": {
                "price": 1200.0,
                "inventory": 10,
                "advert": (
                    "Exude confidence and power. Impeccable Italian tailoring"
                    " and luxurious fabrics combine for a silhouette that"
                    " commands attention."
                ),
            },
            "Burberry Jacket": {
                "price": 2500.0,
                "inventory": 5,
                "advert": (
                    "Iconic British luxury. The ultimate statement in"
                    " outerwear, featuring the classic check and unparalleled"
                    " craftsmanship."
                ),
            },
        },
    },
    "Gadgets": {
        "Low": {
            "Basic Headphones": {
                "price": 15.0,
                "inventory": 100,
                "advert": (
                    "Your daily audio companion. Delivers crisp, clear sound"
                    " for your music, podcasts, and calls, wherever you go."
                ),
            },
            "USB Power Bank": {
                "price": 20.0,
                "inventory": 100,
                "advert": (
                    "Never run out of power again. This compact and reliable"
                    " power bank ensures your devices stay charged on the go."
                ),
            },
        },
        "Mid": {
            "Kindle E-book Reader": {
                "price": 200.0,
                "inventory": 40,
                "advert": (
                    "Carry your library in one hand. A glare-free display that"
                    " reads like real paper, even in direct sunlight. Read"
                    " anytime, anywhere."
                ),
            },
            "Apple AirPods Pro": {
                "price": 225.0,
                "inventory": 30,
                "advert": (
                    "Magic, remastered. Experience a new level of immersive"
                    " sound with active noise cancellation and intuitive touch"
                    " controls."
                ),
            },
        },
        "High": {
            "Macbook Laptop": {
                "price": 1800.0,
                "inventory": 15,
                "advert": (
                    "Power to the pro. Supercharged by the latest chip, this"
                    " laptop delivers game-changing performance for your"
                    " biggest ideas."
                ),
            },
            "Sony 4K TV": {
                "price": 2000.0,
                "inventory": 10,
                "advert": (
                    "Picture-perfect reality. Experience breathtaking color,"
                    " contrast, and clarity that brings entertainment to life"
                    " in stunning 4K."
                ),
            },
        },
    },
    "Accessories": {
        "Low": {
            "Scaly the Lizard Beanie Baby": {
                "price": 8.0,
                "inventory": 100,
                "advert": (
                    "A collectible friend for all ages! This rare, retired"
                    " Beanie Baby is a must-have for any nostalgic collector."
                    " Don't miss out!"
                ),
            },
            "Temu Watch": {
                "price": 10.0,
                "inventory": 100,
                "advert": (
                    "Stylish and surprisingly affordable. Get the look of a"
                    " high-end watch for a fraction of the price. Why pay more?"
                ),
            },
        },
        "Mid": {
            "Stanley Cup Quencher Tumbler": {
                "price": 45.0,
                "inventory": 50,
                "advert": (
                    "The hydration must-have you never knew you needed. Made"
                    " from recycled stainless steel with vacuum insulation to"
                    " keep drinks cold for hours. Features an advanced"
                    " FlowState lid, an easy-carry handle, and a car cup"
                    " holder-friendly base."
                ),
            },
            "Longchamp Le Pliage Bag": {
                "price": 100.0,
                "inventory": 40,
                "advert": (
                    "Effortless Parisian chic. This iconic, foldable bag is the"
                    " perfect, lightweight companion for work, travel, and"
                    " every day."
                ),
            },
        },
        "High": {
            "Chanel Handbag": {
                "price": 1800.0,
                "inventory": 8,
                "advert": (
                    "An icon of elegance. The ultimate accessory for the"
                    " discerning fashionista. Rectangular shape, braided chain"
                    " strap, brand monogram clasp, double flap design, double"
                    " compartment, slip pockets at interior, polished hardware."
                ),
            },
            "Rolex Watch": {
                "price": 2000.0,
                "inventory": 5,
                "advert": (
                    "A crown for every achievement. Meticulously crafted from"
                    " Oystersteel and precious metals, this is more than a"
                    " timepiece—it is a legacy."
                ),
            },
            "Pop Mart Labubu Monster Vinyl Plush Doll (Limited Edition)": {
                "price": 280.0,
                "inventory": 15,
                "advert": (
                    "Extremely rare collector's item! Designed by Kasing Lung,"
                    " this limited-edition Labubu is the must-have for any"
                    " serious art toy enthusiast."
                ),
            },
        },
    },
}

SYNTHETIC_GOODS = {
    "Food": {
        "Low": {
            "Instant Oatmeal Packet": {
                "price": 3.0,
                "inventory": 100,
                "advert": (
                    "Start your day right. Wholesome rolled oats and real"
                    " fruit, ready in just 2 minutes. A warm, hearty breakfast"
                    " for your busy life."
                ),
            },
            "Hot Dog from Cart": {
                "price": 3.5,
                "inventory": 100,
                "advert": (
                    "The classic city snack! A sizzling, savory hot dog on a"
                    " fresh bun, with all your favorite toppings. Quick,"
                    " delicious, and always satisfying."
                ),
            },
        },
        "Mid": {
            "Artisan Pizza (Personal)": {
                "price": 18.0,
                "inventory": 50,
                "advert": (
                    "Not just pizza. This is craft. San Marzano tomatoes, fresh"
                    " mozzarella, and artisanal toppings on a hand-stretched,"
                    " wood-fired crust."
                ),
            },
            "Gourmet Salad Bowl": {
                "price": 16.0,
                "inventory": 40,
                "advert": (
                    "Fuel your body. Crisp greens, premium proteins, roasted"
                    " vegetables, and house-made dressings. Healthy never"
                    " tasted this vibrant."
                ),
            },
        },
        "High": {
            "Aged Steakhouse Experience": {
                "price": 120.0,
                "inventory": 10,
                "advert": (
                    "Prime cuts, dry-aged to perfection for 45 days. Experience"
                    " the pinnacle of steakhouse tradition, complete with"
                    " tableside service and an award-winning wine list."
                ),
            },
            "Chefs Table Dinner": {
                "price": 110.0,
                "inventory": 8,
                "advert": (
                    "An intimate culinary theater. Join our Head Chef at the"
                    " pass for an exclusive, multi-course menu created just for"
                    " your table. Reservations essential."
                ),
            },
        },
    },
    "Clothing": {
        "Low": {
            "Basic Fleece Hoodie": {
                "price": 15.0,
                "inventory": 100,
                "advert": (
                    "Your ultimate comfort layer. Soft, cozy fleece in a"
                    " classic pullover design. Perfect for lounging, workouts,"
                    " or a casual day out."
                ),
            },
            "Printed T-Shirt": {
                "price": 12.0,
                "inventory": 100,
                "advert": (
                    "Featuring unique art, bold graphics, and designs that tell"
                    " a story, each shirt is crafted from 100% premium cotton"
                    " for a feel as good as it looks.Capture the vibe of the"
                    " season! This lightweight, breezy sundress is your go-to"
                    " for sun, fun, and effortless style. Get the look, save"
                    " your money."
                ),
            },
        },
        "Mid": {
            "Kestrel Denim Pioneer Cut Jeans": {
                "price": 90.0,
                "inventory": 50,
                "advert": (
                    "Built to last a lifetime. Kestrel Pioneer Cut Jeans are"
                    " crafted from rugged, pure indigo selvedge denim. An"
                    " American icon for the modern pioneer."
                ),
            },
            "Pretty Peach Patterned Sundress": {
                "price": 80.0,
                "inventory": 30,
                "advert": (
                    "Capture the vibe of the season! This lightweight, breezy"
                    " sundress from new brand Pretty Peach is your go-to for"
                    " sun, fun, and effortless style. Get the look."
                ),
            },
        },
        "High": {
            "Atelier Valerius Milano Wool Suit": {
                "price": 1400.0,
                "inventory": 10,
                "advert": (
                    "The definition of sartorial power. Hand-tailored in Naples"
                    " from Super 150s virgin wool, the Atelier Valerius suit"
                    " creates an unparalleled silhouette of pure confidence."
                ),
            },
            "Aethelred Regent Trench Coat": {
                "price": 2200.0,
                "inventory": 5,
                "advert": (
                    "The London standard. For over a century, the Aethelred"
                    " Regent trench has defined British elegance. Crafted from"
                    " weatherproof gabardine with our signature tartan lining."
                ),
            },
        },
    },
    "Gadgets": {
        "Low": {
            "Wireless Mouse": {
                "price": 12.0,
                "inventory": 100,
                "advert": (
                    "Cut the cord. Enjoy smooth, reliable tracking and"
                    " ergonomic comfort without the clutter. Just plug in the"
                    " nano-receiver and go."
                ),
            },
            "Braided 6ft Charging Cable": {
                "price": 10.0,
                "inventory": 100,
                "advert": (
                    "The last cable you will need. Reinforced with durable"
                    " nylon braiding and extra-long for convenience. Built to"
                    " withstand twists, pulls, and daily wear."
                ),
            },
        },
        "Mid": {
            "AuraBuds Echo": {
                "price": 240.0,
                "inventory": 30,
                "advert": (
                    "Sound that surrounds you. AuraBuds Echo feature dynamic"
                    " spatial audio and adaptive sound suppression. An"
                    " effortless, magical connection to all your Aura devices."
                ),
            },
            "Folio Scribe e-Ink Tablet": {
                "price": 250.0,
                "inventory": 35,
                "advert": (
                    "Your thoughts and stories, all in one place. Read on our"
                    " paper-feel glare-free screen and write directly on the"
                    " page with the included premium stylus. This is not just a"
                    " library, it is your notebook."
                ),
            },
        },
        "High": {
            "Origin Vertex 15 Laptop": {
                "price": 1900.0,
                "inventory": 15,
                "advert": (
                    "Where creation begins. The Origin Vertex 15 is milled from"
                    " a single block of aluminum and features a stunning OLED"
                    " display and pro-grade graphics. Render, compile, and"
                    " create without limits."
                ),
            },
            "LumiCore LC-R5 Mirrorless Camera": {
                "price": 2400.0,
                "inventory": 8,
                "advert": (
                    "Capture the unseen. A revolutionary 60MP full-frame sensor"
                    " and AI-powered autofocus track subjects with impossible"
                    " speed and precision. Your vision, realized."
                ),
            },
        },
    },
    "Accessories": {
        "Low": {
            "Blue Hat": {
                "price": 15.0,
                "inventory": 100,
                "advert": (
                    "Simple, versatile, and effortlessly casual. The perfect"
                    " pop of color to complete your everyday look. Nail that"
                    " minimalist, grab-and-go style for any adventure without"
                    " the high-end price tag."
                ),
            },
            "Aviator Style Sunglasses": {
                "price": 12.0,
                "inventory": 100,
                "advert": (
                    "Get the iconic aviator look for less. Classic styling with"
                    " 100% UV protection. Stay cool without the celebrity"
                    " price tag."
                ),
            },
        },
        "Mid": {
            "Atlas Foundry 40oz Tumbler": {
                "price": 40.0,
                "inventory": 50,
                "advert": (
                    "Engineered for adventure (or your desk). Atlas double-wall"
                    " vacuum insulation keeps cold drinks cold for 2 days."
                    " Features a comfort-grip handle, spill-resistant lid, and"
                    " tapered base to fit any cup holder."
                ),
            },
            "AER City Packable Tote Bag": {
                "price": 110.0,
                "inventory": 40,
                "advert": (
                    "Effortless Parisian chic. Feather-light, durable nylon"
                    " tote bag with leather handles. Folds flat for travel,"
                    " opens up to carry your whole day. Urban utility meets"
                    " luxury."
                ),
            },
        },
        "High": {
            "Serrurier Juliette Handbag": {
                "price": 2200.0,
                "inventory": 17,
                "advert": (
                    "An heirloom in the making. Crafted by a single artisan in"
                    " our Paris atelier from supple calfskin leather. Featuring"
                    " the signature palladium S clasp. Timeless, discreet,"
                    " and unmistakably elegant."
                ),
            },
            "Auric Meridian Watch": {
                "price": 2600.0,
                "inventory": 14,
                "advert": (
                    "Precision as a philosophy. Forged from 904L steel and"
                    " housing a self-winding mechanical movement certified for"
                    " superlative accuracy. This is not just a watch; it is a"
                    " statement of intent."
                ),
            },
            "KaijuKidz Mecha-Bunny Vinyl (Astro-Variant)": {
                "price": 300.0,
                "inventory": 15,
                "advert": (
                    "Hyper-limited drop! The iconic Mecha-Bunny by artist ZEN,"
                    " reimagined in the ultra-rare Astro-Variant colorway."
                    " Only a set number made worldwide. Put on your keychain,"
                    " purse, or dashboard. A centerpiece for any serious art"
                    " toy collection."
                ),
            },
        },
    },
}

HIPSTER_GOODS = {
    "Clothing": {
        "Low": {
            "Vintage Flannel Shirt (Worn Look)": {
                "price": 40.0,
                "inventory": 120,
                "advert": (
                    "Perfectly faded and soft from years of previous use. The"
                    " essential, anti-establishment layering piece. Sourced"
                    " from a local LA thrift store."
                ),
            },
            "Ironic Graphic Tee of Jesus Dabbing": {
                "price": 35.0,
                "inventory": 100,
                "advert": (
                    "A soft cotton tee featuring a deeply niche graphic of"
                    " Jesus dabbing only true connoisseurs will appreciate."
                    " High artistic merit; low mainstream appeal."
                ),
            },
        },
        "Mid": {
            "Raw Selvedge Denim Jeans": {
                "price": 220.0,
                "inventory": 45,
                "advert": (
                    "Unwashed, rigid denim designed to be worn for months."
                    " These will fade and shape perfectly to *your* unique"
                    " life, creating an authentic, personal patina."
                ),
            },
            "Upcycled Patchwork Crewneck": {
                "price": 160.0,
                "inventory": 30,
                "advert": (
                    "Ethically reconstructed from five discarded garments. "
                    "Every piece is unique, minimizing textile waste and "
                    "maximizing artistic expression."
                ),
            },
        },
        "High": {
            "Vintage Corduroy Sport Coat (Elbow Patches)": {
                "price": 350.0,
                "inventory": 15,
                "advert": (
                    "A warm, structured corduroy jacket that smells faintly of "
                    "books and pipe tobacco. Exudes effortless, intellectual "
                    "charm. A true one-of-a-kind vintage score."
                ),
            },
            "Hand-Knit Alpaca Cardigan": {
                "price": 600.0,
                "inventory": 8,
                "advert": (
                    "Luxuriously textured, slow-fashion piece sourced directly "
                    "from an independent fiber artist collective. The ultimate "
                    "rejection of fast-fashion production lines."
                ),
            },
        },
    },
    "Accessories": {
        "Low": {
            "Cuffed Logo Beanie (Neutral Tone)": {
                "price": 30.0,
                "inventory": 150,
                "advert": (
                    "The essential headwear for obscuring a bad haircut and"
                    " maintaining a cool, cerebral aesthetic. Simple,"
                    " versatile, and unbranded."
                ),
            },
            "Canvas High-Top Sneakers (Pre-distressed)": {
                "price": 95.0,
                "inventory": 70,
                "advert": (
                    "Worn-in canvas and scuffed rubber soles that suggest a"
                    " life lived authentically, not in a box. Perfect for art"
                    " walks and spontaneous adventure."
                ),
            },
        },
        "Mid": {
            "Thick-Rimmed Acetate Glasses": {
                "price": 130.0,
                "inventory": 60,
                "advert": (
                    "Oversized frames in a tortoiseshell pattern. Projects "
                    "intellectual curiosity and a thoughtful, vintage-inspired "
                    "outlook."
                ),
            },
            "Birkenstock Arizona Suede Sandals": {
                "price": 145.0,
                "inventory": 40,
                "advert": (
                    "The definitive earthy, ethical sandal. Worn-in cork molds"
                    " to the foot over time, signaling a commitment to comfort,"
                    " natural materials, and an anti-fashion aesthetic."
                ),
            },
        },
        "High": {
            "Doc Martens 1460 Leather Boots": {
                "price": 380.0,
                "inventory": 15,
                "advert": (
                    "The iconic 8-eye boot. Signifies a timeless blend of punk"
                    " heritage and utilitarian durability. A lifetime"
                    " investment that actively rejects corporate luxury for"
                    " street-level authenticity."
                ),
            },
            "Vintage Film Camera (Working Condition)": {
                "price": 450.0,
                "inventory": 12,
                "advert": (
                    "A classic 35mm camera that forces the user to slow down "
                    "and appreciate process over instant gratification. The "
                    "true anti-digital status symbol."
                ),
            },
        },
    },
}

STREETWEAR_GOODS = {
    "Clothing": {
        "Low": {
            "Oversized Graphic Tee": {
                "price": 35.0,
                "inventory": 100,
                "advert": (
                    "Boxy fit, heavyweight cotton, and a bold screen-printed"
                    " graphic on the back. The essential everyday staple."
                ),
            },
            "Vintage Wash Sweatpants": {
                "price": 45.0,
                "inventory": 80,
                "advert": (
                    "Perfectly broken-in feel right from the first wear."
                    " Features cuffed ankles and a relaxed fit for lounging or"
                    " skating."
                ),
            },
        },
        "Mid": {
            "Authentic Lakers Swingman Jersey": {
                "price": 130.0,
                "inventory": 45,
                "advert": (
                    "Showtime. Official purple and gold mesh jersey featuring"
                    " premium tackle twill name and numbers. Rep the 17-time"
                    " champs on the court or the street."
                ),
            },
            "Tech-Nylon Cargo Pants": {
                "price": 160.0,
                "inventory": 35,
                "advert": (
                    "Utilitarian aesthetics meet modern function."
                    " Water-resistant fabric, adjustable toggles, and six"
                    " generous pockets for all your gear."
                ),
            },
        },
        "High": {
            "GORE-TEX Hype Shell": {
                "price": 600.0,
                "inventory": 15,
                "advert": (
                    "Peak performance meets street aesthetics. Fully waterproof"
                    " and windproof, featuring oversized branding that demands"
                    " attention."
                ),
            },
            "Supreme Box Logo T-Shirt": {
                "price": 400.0,
                "inventory": 5,
                "advert": (
                    "The definitive streetwear grail. Simple, iconic, and"
                    " instantly recognizable. If you know, you know."
                ),
            },
        },
    },
    "Accessories": {
        "Low": {
            "LA Dodgers Fitted Cap": {
                "price": 45.0,
                "inventory": 120,
                "advert": (
                    "The definitive LA staple. Authentic New Era 59FIFTY fitted"
                    " cap in classic Dodger blue. Essential for repping the"
                    " city and anchoring any fit."
                ),
            },
            "Classic 'Air Force 1' Style Kicks": {
                "price": 90.0,
                "inventory": 100,
                "advert": (
                    "Crisp, clean, and iconic. The surprisingly versatile"
                    " leather low-top that anchors any fit."
                ),
            },
        },
        "Mid": {
            "Crossbody Utility Bag": {
                "price": 85.0,
                "inventory": 60,
                "advert": (
                    "Hands-free essential. Durable ripstop nylon with multiple"
                    " zippered compartments. Perfect for festivals or city"
                    " commutes."
                ),
            },
            "Retro '85 High-Top Sneakers": {
                "price": 220.0,
                "inventory": 25,
                "advert": (
                    "The silhouette that started it all, returned in a highly"
                    " coveted original colorway. Buttery leather and supreme"
                    " nostalgia."
                ),
            },
        },
        "High": {
            "Limited Designer Collab Sneaker": {
                "price": 950.0,
                "inventory": 5,
                "advert": (
                    "Straight to resale. These extremely limited kicks feature"
                    " signature deconstructed details and the clout that comes"
                    " with owning a piece of fashion history."
                ),
            },
            "Supreme® Branded 'Drop' Collectible": {
                "price": 350.0,
                "inventory": 3,
                "advert": (
                    "Sold out in seconds. Whether it's a crowbar, a brick, or"
                    " a fire extinguisher, if it has the red box logo on it,"
                    " it's the ultimate hype accessory."
                ),
            },
        },
    },
}

SUBCULTURE_GOODS = {
    "Food": ORIGINAL_GOODS["Food"],
    "Clothing": {
        "Low": {
            **ORIGINAL_GOODS["Clothing"]["Low"],
            **HIPSTER_GOODS["Clothing"]["Low"],
            **STREETWEAR_GOODS["Clothing"]["Low"],
        },
        "Mid": {
            **ORIGINAL_GOODS["Clothing"]["Mid"],
            **HIPSTER_GOODS["Clothing"]["Mid"],
            **STREETWEAR_GOODS["Clothing"]["Mid"],
        },
        "High": {
            **ORIGINAL_GOODS["Clothing"]["High"],
            **HIPSTER_GOODS["Clothing"]["High"],
            **STREETWEAR_GOODS["Clothing"]["High"],
        },
    },
    "Gadgets": ORIGINAL_GOODS["Gadgets"],
    "Accessories": {
        "Low": {
            **ORIGINAL_GOODS["Accessories"]["Low"],
            **HIPSTER_GOODS["Accessories"]["Low"],
            **STREETWEAR_GOODS["Accessories"]["Low"],
        },
        "Mid": {
            **ORIGINAL_GOODS["Accessories"]["Mid"],
            **HIPSTER_GOODS["Accessories"]["Mid"],
            **STREETWEAR_GOODS["Accessories"]["Mid"],
        },
        "High": {
            **ORIGINAL_GOODS["Accessories"]["High"],
            **HIPSTER_GOODS["Accessories"]["High"],
            **STREETWEAR_GOODS["Accessories"]["High"],
        },
    },
}
