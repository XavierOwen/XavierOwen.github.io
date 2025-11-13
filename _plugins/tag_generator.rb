module Jekyll
  class TagPageGenerator < Generator
    safe true

    def generate(site)
      if site.layouts.key? 'tag'
        dir = site.config['tag_dir'] || 'tags'

        # Get all tags from all collections and posts
        all_tags = Set.new

        # Process posts
        site.posts.docs.each do |post|
          post.data['tags']&.each { |tag| all_tags.add(tag) }
        end

        # Process collections
        site.collections.each do |_, collection|
          collection.docs.each do |doc|
            doc.data['tags']&.each { |tag| all_tags.add(tag) }
          end
        end

        # Generate a page for each tag
        all_tags.each do |tag|
          site.pages << TagPage.new(site, site.source, File.join(dir, tag), tag)
        end
      end
    end
  end

  # A Page subclass used in the `TagPageGenerator`
  class TagPage < Page
    def initialize(site, base, dir, tag)
      @site = site
      @base = base
      @dir  = dir
      @name = 'index.html'

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'tag.html')
      self.data['tag'] = tag
      self.data['title'] = "Tagged: #{tag}"
    end
  end
end