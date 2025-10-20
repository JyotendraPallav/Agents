// frontend/app/page.tsx
import Link from 'next/link';
// Import our new FilterBar component
import FilterBar from '@/components/FilterBar';

// --- Define our data types (same as before) ---
interface Book {
  id: number;
  title: string;
  authors: string[];
  goodreads_rating: number;
  genre: string;
  image_url: string | null;
}

// --- getBooks function (unchanged) ---
async function getBooks(search: string, genre: string) {
  try {
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (genre) params.set('genre', genre);
    
    const res = await fetch(`http://127.0.0.1:8000/api/books?${params.toString()}`, { 
      cache: 'no-store'
    });
    
    if (!res.ok) {
      throw new Error('Failed to fetch books from the API');
    }
    
    const data = await res.json();
    return data.books as Book[];
  } catch (error) {
    console.error(error);
    return [];
  }
}

// --- getGenres function (unchanged) ---
async function getGenres() {
  try {
    const res = await fetch('http://127.0.0.1:8000/api/genres', { cache: 'no-store' });
    if (!res.ok) { throw new Error('Failed to fetch genres'); }
    const data = await res.json();
    return data.genres as string[];
  } catch (error) {
    console.error(error);
    return [];
  }
}

// --- Updated Homepage Component ---
export default async function HomePage({
  searchParams
}: {
  searchParams?: {
    search?: string;
    genre?: string;
  };
}) {
  const currentSearch = searchParams?.search || '';
  const currentGenre = searchParams?.genre || '';

  // Fetch both books and genres at the same time
  const books = await getBooks(currentSearch, currentGenre);
  const genres = await getGenres();

  return (
    <main className="bg-gray-50 min-h-screen">
      <div className="container mx-auto p-4 sm:p-8">
        <header className="text-center mb-10">
          <h1 className="text-5xl font-bold text-gray-800">LibrisAI</h1>
          <p className="text-lg text-gray-500 mt-2">Your AI-Curated Digital Bookshelf</p>
        </header>

        {/* --- NEW: Filter Bar --- */}
        {/* We pass the genres list from the server to the client component */}
        <div className="mb-10">
          <FilterBar genres={genres} />
        </div>

        {/* --- The Genre Filter Buttons are now GONE --- */}
        
        {/* The grid that holds all our book cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">
          
          {/* Book Card Loop (unchanged) */}
          {books.map((book) => (
            <Link href={`/${book.id}`} key={book.id} className="flex">
              <div className="bg-white border rounded-lg shadow-md overflow-hidden transform hover:-translate-y-1 transition-transform duration-300 flex flex-col w-full">
                
                {/* --- IMAGE BLOCK (unchanged) --- */}
                <div className="h-80 w-full relative">
                  {book.image_url ? (
                    <img
                      src={book.image_url}
                      alt={book.title}
                      className="object-cover w-full h-full"
                    />
                  ) : (
                    <div className="h-full w-full bg-gray-200 flex items-center justify-center">
                      <span className="text-gray-400 text-sm">No Cover Image</span>
                    </div>
                  )}
                </div>
                
                {/* Book Info Area (unchanged) */}
                <div className="p-4 flex-grow">
                  <h2 className="text-md font-bold text-gray-800 truncate" title={book.title}>
                    {book.title}
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">
                    {book.authors ? book.authors.join(', ') : 'Unknown Author'}
                  </p>
                  <div className="flex justify-between items-center mt-4">
                    <span className="text-xs font-semibold bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                      {book.genre || 'Uncategorized'}
                    </span>
                    <span className="font-bold text-gray-700 flex items-center">
                      <svg className="w-4 h-4 text-yellow-400 mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
                      {book.goodreads_rating}
                    </span>
                  </div>
                </div>
              </div>
            </Link>
          ))}

          {/* "No books found" message (unchanged) */}
          {books.length === 0 && (
            <div className="col-span-full text-center text-gray-500 py-10">
              <h2 className="text-2xl font-semibold">No books found</h2>
              <p>Try adjusting your search or filters.</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}