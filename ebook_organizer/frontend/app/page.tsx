// This is your new frontend/app/page.tsx file

// This defines what a "Book" object looks like
interface Book {
  id: number;
  title: string;
  authors: string[];
  goodreads_rating: number;
  genre: string;
  image_url: string | null; // The image might be missing for some books
}

// This function runs on the server to fetch data from our Python backend
async function getBooks() {
  try {
    // We fetch from the backend API we have running on port 8000
    const res = await fetch('http://127.0.0.1:8000/api/books', { 
      cache: 'no-store' // Ensures we always get the freshest data
    });

    if (!res.ok) {
      throw new Error('Failed to fetch books from the API');
    }

    const data = await res.json();
    return data.books as Book[];
  } catch (error) {
    console.error(error);
    return []; // Return an empty list if there's an error
  }
}

// This is the main component for our homepage
export default async function HomePage() {
  const books = await getBooks();

  return (
    <main className="bg-gray-50 min-h-screen">
      <div className="container mx-auto p-4 sm:p-8">
        <header className="text-center mb-10">
          <h1 className="text-5xl font-bold text-gray-800">LibrisAI</h1>
          <p className="text-lg text-gray-500 mt-2">Your AI-Curated Digital Bookshelf</p>
        </header>

        {/* The grid that holds all our book cards */}
        <div className="grid grid-cols-1 sm:g:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">

          {/* We loop over each book and create a card for it */}
          {books.map((book) => (
            <div key={book.id} className="bg-white border rounded-lg shadow-md overflow-hidden transform hover:-translate-y-1 transition-transform duration-300">

              {/* Image Placeholder - we will fix this in the next phase! */}
              <div className="h-64 bg-gray-200 flex items-center justify-center">
                <span className="text-gray-400 text-sm">No Cover Image</span>
              </div>

              {/* Book Info Area */}
              <div className="p-4">
                <h2 className="text-md font-bold text-gray-800 truncate" title={book.title}>
                  {book.title}
                </h2>

                {/* Handle authors (it's a list) */}
                <p className="text-sm text-gray-600 mt-1">
                  {book.authors ? book.authors.join(', ') : 'Unknown Author'}
                </p>

                <div className="flex justify-between items-center mt-4">
                  <span className="text-xs font-semibold bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                    {book.genre || 'Uncategorized'}
                  </span>
                  <span className="font-bold text-gray-700 flex items-center">
                    {/* Star Icon */}
                    <svg className="w-4 h-4 text-yellow-400 mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
                    {book.goodreads_rating}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}