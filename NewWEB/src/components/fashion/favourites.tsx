'use client';

import React from 'react';
import { Heart, ShoppingBag, Star } from 'lucide-react';

interface ClothingItem {
  id: string;
  name: string;
  category: string;
  description: string;
  colors: string[];
  material: string;
  style: string;
  fit: string;
  sleeves?: string;
  neckline?: string;
  length?: string;
  details?: string;
  price: string;
  rating: number;
  imageUrl: string;
}

const favouriteItems: ClothingItem[] = [
  {
    id: "00165_00",
    name: "Classic Black Polo",
    category: "Polo Shirts",
    description: "Casual black polo shirt with lettering print and buttoned collar detail",
    colors: ["Black"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Short Sleeve",
    neckline: "Shirt Collar",
    length: "Normal",
    details: "Buttoned",
    price: "$45",
    rating: 4.8,
    imageUrl: "/cloth/00165_00.jpg"
  },
  {
    id: "00235_00",
    name: "Vibrant Yellow Tee",
    category: "T-shirts",
    description: "Drop shoulder yellow v-neck t-shirt in soft cotton fabric",
    colors: ["Yellow"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Long Sleeve",
    neckline: "V Neck",
    length: "Normal",
    details: "Drop shoulder",
    price: "$32",
    rating: 4.6,
    imageUrl: "/cloth/00235_00.jpg"
  },
  {
    id: "00619_00",
    name: "Sheer White Basic",
    category: "T-shirts",
    description: "Semi-transparent white knit t-shirt with round neckline",
    colors: ["White"],
    material: "Knit",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Short Sleeve",
    neckline: "Round Neck",
    length: "Normal",
    details: "See through",
    price: "$28",
    rating: 4.4,
    imageUrl: "/cloth/00619_00.jpg"
  },
  {
    id: "00825_00",
    name: "Purple V-Neck Tee",
    category: "T-shirts",
    description: "Drop shoulder purple cotton tee with comfortable v-neck design",
    colors: ["Purple"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Short Sleeve",
    neckline: "V Neck",
    length: "Normal",
    details: "Drop shoulder",
    price: "$30",
    rating: 4.7,
    imageUrl: "/cloth/00825_00.jpg"
  },
  {
    id: "00834_00",
    name: "White Frilled Bra Top",
    category: "Bra Top",
    description: "Feminine white cotton bra top with frill details and v-neck styling",
    colors: ["White"],
    material: "Cotton",
    style: "Feminine",
    fit: "Tight Fit",
    sleeves: "Sleeveless",
    neckline: "V Neck",
    length: "Cropped",
    details: "Frill",
    price: "$35",
    rating: 4.5,
    imageUrl: "/cloth/00834_00.jpg"
  },
  {
    id: "00912_00",
    name: "Orange Drop Shoulder",
    category: "T-shirts",
    description: "Bright orange cotton tee with relaxed drop shoulder design",
    colors: ["Orange"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Short Sleeve",
    neckline: "Round Neck",
    length: "Normal",
    details: "Drop shoulder",
    price: "$29",
    rating: 4.6,
    imageUrl: "/cloth/00912_00.jpg"
  },
  {
    id: "01024_00",
    name: "Black Graphic Long Sleeve",
    category: "T-shirts",
    description: "Edgy black synthetic tee with graphic print and stitched details",
    colors: ["Black"],
    material: "Synthetic",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Long Sleeve",
    neckline: "Round Neck",
    length: "Normal",
    details: "Stitch",
    price: "$38",
    rating: 4.3,
    imageUrl: "/cloth/01024_00.jpg"
  },
  {
    id: "01684_00",
    name: "Beige Striped Tank",
    category: "Tank Top",
    description: "Casual beige cotton tank top with stripe pattern and button details",
    colors: ["Beige"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Sleeveless",
    neckline: "V Neck",
    length: "Normal",
    details: "Buttoned",
    price: "$26",
    rating: 4.5,
    imageUrl: "/cloth/01684_00.jpg"
  },
  {
    id: "02842_00",
    name: "Black Drop Shoulder Sweatshirt",
    category: "Sweatshirt",
    description: "Cozy black cotton sweatshirt with oversized drop shoulder cut",
    colors: ["Black"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Long Sleeve",
    neckline: "Round Neck",
    length: "Normal",
    details: "Drop shoulder",
    price: "$55",
    rating: 4.9,
    imageUrl: "/cloth/02842_00.jpg"
  },
  {
    id: "02974_00",
    name: "Dotted Black Sweater",
    category: "Sweater",
    description: "Oversized black knit sweater with polka dot pattern and v-neck",
    colors: ["Black"],
    material: "Knit",
    style: "Casual",
    fit: "Oversized",
    sleeves: "Long Sleeve",
    neckline: "V Neck",
    length: "Normal",
    details: "Drop shoulder",
    price: "$65",
    rating: 4.7,
    imageUrl: "/cloth/02974_00.jpg"
  },
  {
    id: "03452_00",
    name: "Lavender Lace Bustier",
    category: "Bustier",
    description: "Elegant lavender lace bustier with see-through details and v-neck",
    colors: ["Lavender"],
    material: "Lace",
    style: "Feminine",
    fit: "Tight Fit",
    sleeves: "Sleeveless",
    neckline: "V Neck",
    length: "Cropped",
    details: "See through",
    price: "$48",
    rating: 4.6,
    imageUrl: "/cloth/03452_00.jpg"
  },
  {
    id: "03643_00",
    name: "Green Striped Blouse",
    category: "Blouse",
    description: "Ethnic-inspired green blouse with stripe pattern and frill details",
    colors: ["Green"],
    material: "Synthetic",
    style: "Ethnic",
    fit: "Normal Fit",
    sleeves: "Sleeveless",
    neckline: "Round Neck",
    length: "Normal",
    details: "Frill",
    price: "$42",
    rating: 4.4,
    imageUrl: "/cloth/03643_00.jpg"
  },
  {
    id: "05209_00",
    name: "Black Pleated Skirt",
    category: "Pleats Skirt",
    description: "Classic black synthetic pleated skirt with knee-length H-line silhouette",
    colors: ["Black"],
    material: "Synthetic",
    style: "Casual",
    fit: "H-line",
    length: "Knee Length",
    details: "Pleats",
    price: "$38",
    rating: 4.8,
    imageUrl: "/cloth/05209_00.jpg"
  },
  {
    id: "11647_00",
    name: "Black Mermaid Skirt",
    category: "Trumpet Skirt",
    description: "Elegant black synthetic trumpet skirt with ruffle details and mermaid line",
    colors: ["Black"],
    material: "Synthetic",
    style: "Feminine",
    fit: "Mermaid Line",
    length: "Long",
    details: "Ruffle",
    price: "$52",
    rating: 4.7,
    imageUrl: "/cloth/11647_00.jpg"
  },
  {
    id: "02175_00",
    name: "Black Paisley Tunic",
    category: "Tunic Dress",
    description: "Feminine black synthetic tunic dress with paisley print pattern",
    colors: ["Black"],
    material: "Synthetic",
    style: "Feminine",
    fit: "H-line",
    sleeves: "Short Sleeve",
    neckline: "Round Neck",
    length: "Mini",
    price: "$45",
    rating: 4.5,
    imageUrl: "/cloth/02175_00.jpg"
  },
  {
    id: "04023_00",
    name: "Skull Print Swimsuit",
    category: "One piece Swimsuit",
    description: "Resort-style black spandex swimsuit with skull print and stitched details",
    colors: ["Black"],
    material: "Spandex",
    style: "Resort",
    fit: "Fitted",
    sleeves: "Long Sleeve",
    neckline: "Round Neck",
    details: "Stitch",
    price: "$58",
    rating: 4.3,
    imageUrl: "/cloth/04023_00.jpg"
  },
  {
    id: "10257_00",
    name: "Purple Zip Swimsuit",
    category: "One piece Swimsuit",
    description: "Sporty purple spandex swimsuit with zip-up front and stand-up collar",
    colors: ["Purple"],
    material: "Spandex",
    style: "Resort",
    fit: "Athletic",
    sleeves: "Long Sleeve",
    neckline: "Stand-up Collar",
    details: "Zip up",
    price: "$62",
    rating: 4.6,
    imageUrl: "/cloth/10257_00.jpg"
  },
  {
    id: "10276_00",
    name: "Floral A-line Dress",
    category: "Tunic Dress",
    description: "Feminine black dress with floral print, shirring details and A-line silhouette",
    colors: ["Black"],
    material: "Synthetic",
    style: "Feminine",
    fit: "A-line",
    sleeves: "Short Sleeve",
    neckline: "Round Neck",
    length: "Knee Length",
    details: "Shirring",
    price: "$48",
    rating: 4.8,
    imageUrl: "/cloth/10276_00.jpg"
  },
  {
    id: "02695_00",
    name: "Sky Blue Graphic Polo",
    category: "Polo Shirts",
    description: "Casual sky blue cotton polo with graphic print and shirt collar",
    colors: ["Sky Blue"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Short Sleeve",
    neckline: "Shirt Collar",
    length: "Normal",
    details: "Buttoned",
    price: "$44",
    rating: 4.5,
    imageUrl: "/cloth/02695_00.jpg"
  },
  {
    id: "03706_00",
    name: "Navy Embroidered Tee",
    category: "T-shirts",
    description: "Navy cotton t-shirt with embroidered lettering and round neckline",
    colors: ["Navy"],
    material: "Cotton",
    style: "Casual",
    fit: "Normal Fit",
    sleeves: "Short Sleeve",
    neckline: "Round Neck",
    length: "Normal",
    details: "Embroidery",
    price: "$36",
    rating: 4.7,
    imageUrl: "/cloth/03706_00.jpg"
  }
];

const ClothingCard: React.FC<{ item: ClothingItem }> = ({ item }) => {
  return (
    <div className="bg-white rounded-xl shadow-md hover:shadow-lg transition-all duration-300 overflow-hidden group">
      <div className="relative overflow-hidden">
        <img
          src={item.imageUrl}
          alt={item.name}
          className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
          onError={(e) => {
            e.currentTarget.src = '/placeholder-cloth.png';
          }}
        />
        <div className="absolute top-3 right-3">
          <button className="bg-white/90 p-2 rounded-full hover:bg-white transition-colors">
            <Heart className="w-4 h-4 text-red-500 fill-current" />
          </button>
        </div>
        <div className="absolute top-3 left-3">
          <span className="bg-black/80 text-white px-2 py-1 rounded-md text-xs font-medium">
            {item.category}
          </span>
        </div>
      </div>
      
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-gray-900 text-sm">{item.name}</h3>
          <span className="text-lg font-bold text-indigo-600">{item.price}</span>
        </div>
        
        <p className="text-gray-600 text-xs mb-3 line-clamp-2">{item.description}</p>
        
        <div className="flex items-center mb-3">
          <div className="flex items-center">
            {[...Array(5)].map((_, i) => (
              <Star
                key={i}
                className={`w-3 h-3 ${
                  i < Math.floor(item.rating)
                    ? 'text-yellow-400 fill-current'
                    : 'text-gray-300'
                }`}
              />
            ))}
          </div>
          <span className="ml-1 text-xs text-gray-500">({item.rating})</span>
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-3">
          <div>
            <span className="font-medium">Material:</span> {item.material}
          </div>
          <div>
            <span className="font-medium">Fit:</span> {item.fit}
          </div>
          {item.sleeves && (
            <div>
              <span className="font-medium">Sleeves:</span> {item.sleeves}
            </div>
          )}
          {item.details && (
            <div>
              <span className="font-medium">Details:</span> {item.details}
            </div>
          )}
        </div>
        
        <div className="flex flex-wrap gap-1 mb-3">
          {item.colors.map((color, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs"
            >
              {color}
            </span>
          ))}
        </div>
        
        <button className="w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2 text-sm">
          <ShoppingBag className="w-4 h-4" />
          Add to Cart
        </button>
      </div>
    </div>
  );
};

const Favourites: React.FC = () => {
  return (
    <section className="py-16 bg-gradient-to-br from-purple-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            ✨ Our Favourite Picks
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Curated collection of our most loved clothing pieces - from casual everyday wear 
            to elegant statement pieces that define your style
          </p>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
          {favouriteItems.map((item) => (
            <ClothingCard key={item.id} item={item} />
          ))}
        </div>
        
        <div className="text-center mt-12">
          <button className="bg-indigo-600 text-white px-8 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-medium">
            View All Collection
          </button>
        </div>
      </div>
    </section>
  );
};

export default Favourites;
