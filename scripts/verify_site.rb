#!/usr/bin/env ruby
# frozen_string_literal: true

require "open3"

ROOT = File.expand_path("..", __dir__)
COMMANDS = [
  ["ruby", "scripts/check_pages_workflow.rb"],
  ["ruby", "scripts/audit_published_content.rb"],
  ["ruby", "scripts/build_content_index.rb"],
  ["bundle", "exec", "jekyll", "build"],
  ["ruby", "scripts/check_static_assets.rb"],
  ["ruby", "scripts/check_reader_paths.rb"],
  ["ruby", "scripts/check_bilingual_articles.rb"],
  ["ruby", "scripts/check_wiki_index.rb"],
  ["ruby", "scripts/check_manifesto_home.rb"],
  ["node", "scripts/test_language_preference.js"],
  ["node", "scripts/test_wiki_links.js"],
  ["node", "--check", "assets/js/language-preference.js"],
  ["node", "--check", "assets/js/toc-scrollspy.js"],
  ["node", "--check", "assets/js/wiki-links.js"],
  ["git", "diff", "--check"]
].freeze

COMMANDS.each do |command|
  puts "\n==> #{command.join(" ")}"
  status = nil
  Open3.popen2e(*command, chdir: ROOT) do |_stdin, output, wait_thread|
    output.each { |line| print line }
    status = wait_thread.value
  end
  next if status.success?

  warn "verify:site failed: #{command.join(" ")}"
  exit status.exitstatus
end

puts "\nverify:site passed."
