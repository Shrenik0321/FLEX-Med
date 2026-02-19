export default function Footer() {
  return (
    <footer className="bg-background text-foreground">
      <div className="container mx-auto px-4 py-6">
        <p className="text-center text-sm text-gray-500">
          &copy; {new Date().getFullYear()} FLEX-Med. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
